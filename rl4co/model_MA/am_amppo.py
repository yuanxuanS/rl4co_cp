from functools import partial
from typing import Any, Iterable, Union

import torch
import torch.nn as nn

from lightning import LightningModule
from rl4co.model_MA.base import RL4COMarlLitModule
from torch.utils.data import DataLoader

from rl4co.heuristic import CW_svrp, TabuSearch_svrp, Random_svrp
from rl4co.models.zoo.am import AttentionModel
from rl4co.model_adversary import PPOContiAdvModel


from rl4co.data.dataset import tensordict_collate_fn
from rl4co.data.generate_data import generate_default_datasets
from rl4co.envs.common.base import RL4COEnvBase
from rl4co.utils.optim_helpers import create_optimizer, create_scheduler
from rl4co.utils.pylogger import get_pylogger

from lightning.pytorch.utilities import grad_norm

from memory_profiler import profile
import time
log = get_pylogger(__name__)

OPPONENTS_REGISTRY = {
    "cw": CW_svrp,
    "tabu": TabuSearch_svrp,
    "random": Random_svrp
}

ADVERSARY_REGISTRY = {
    "am-ppo": PPOContiAdvModel,
}

PROTAGONIST_REGISTRY = {
    "am": AttentionModel,
}

class AM_PPO(RL4COMarlLitModule):
    """Base class for "Adversary" Lightning modules for RL4CO. This defines the general training loop in terms of
    Adversary RL algorithms. Subclasses should implement mainly the `shared_step` to define the specific
    loss functions and optimization routines.
    
    As an advesary policy to interact, main policy get reward, for advesary is -reward.
    
    Args:
        env: RL4CO environment
        policy: policy network (actor)
        batch_size: batch size (general one, default used for training)
        val_batch_size: specific batch size for validation
        test_batch_size: specific batch size for testing
        train_data_size: size of training dataset for one epoch
        val_data_size: size of validation dataset for one epoch
        test_data_size: size of testing dataset for one epoch
        optimizer: optimizer or optimizer name
        optimizer_kwargs: optimizer kwargs
        lr_scheduler: learning rate scheduler or learning rate scheduler name
        lr_scheduler_kwargs: learning rate scheduler kwargs
        lr_scheduler_interval: learning rate scheduler interval
        lr_scheduler_monitor: learning rate scheduler monitor
        generate_data: whether to generate data
        shuffle_train_dataloader: whether to shuffle training dataloader
        dataloader_num_workers: number of workers for dataloader
        data_dir: data directory
        metrics: metrics
        litmodule_kwargs: kwargs for `LightningModule`
    """

    def __init__(
        self,
        env: RL4COEnvBase,
        protagonist: Union[str, LightningModule] = None,
        adversary: Union[str, LightningModule] = None,
        batch_size: int = 512,
        val_batch_size: int = None,
        test_batch_size: int = None,
        train_data_size: int = 1_280_000,
        val_data_size: int = 10_000,
        test_data_size: int = 10_000,
        optimizer: Union[str, torch.optim.Optimizer, partial] = "Adam",
        optimizer_kwargs: dict = {"lr": 1e-4},
        lr_scheduler: Union[str, torch.optim.lr_scheduler.LRScheduler, partial] = None,
        lr_scheduler_kwargs: dict = {
            "milestones": [80, 95],
            "gamma": 0.1,
        },
        lr_scheduler_interval: str = "epoch",
        lr_scheduler_monitor: str = "val/reward",
        generate_data: bool = True,
        shuffle_train_dataloader: bool = True,
        dataloader_num_workers: int = 0,
        data_dir: str = "data/",
        log_on_step: bool = True,
        adv_log_on_step: bool = True,
        metrics: dict = {},
        **litmodule_kwargs,
    ):
        super().__init__(env, protagonist, adversary, **litmodule_kwargs)

        # This line ensures params passed to LightningModule will be saved to ckpt
        # it also allows to access params with 'self.hparams' attribute
        # Note: we will send to logger with `self.logger.save_hyperparams` in `setup`
        self.save_hyperparameters(logger=False)

        self.env = env
        if isinstance(protagonist, str):
            protagonist = get_protagonist(protagonist)
        self.protagonist = protagonist
        self.protagonist.automatic_optimization = False
        
        if isinstance(adversary, str):
            adversary = get_adversary(adversary)
        self.adversary = adversary
        
        # 加
        self.automatic_optimization = False

        self.instantiate_metrics(metrics)
        self.log_on_step = log_on_step
        self.adv_log_on_step = adv_log_on_step

        self.data_cfg = {
            "batch_size": batch_size,
            "val_batch_size": val_batch_size,
            "test_batch_size": test_batch_size,
            "generate_data": generate_data,
            "data_dir": data_dir,
            "train_data_size": train_data_size,
            "val_data_size": val_data_size,
            "test_data_size": test_data_size,
        }

        self._optimizer_name_or_cls = None
        self.optimizer_kwargs = None
        self._lr_scheduler_name_or_cls = None
        self.lr_scheduler_kwargs = None
        self.lr_scheduler_interval = None
        self.lr_scheduler_monitor = None

        self.shuffle_train_dataloader = shuffle_train_dataloader
        self.dataloader_num_workers = dataloader_num_workers

    def instantiate_metrics(self, metrics: dict):
        """Dictionary of metrics to be logged at each phase"""

        if not metrics:
            log.info("No metrics specified, using default")
        self.train_metrics = metrics.get("train", ["loss", "reward"])
        self.val_metrics = metrics.get("val", ["reward"])
        self.test_metrics = metrics.get("test", ["reward"])
        self.log_on_step = metrics.get("log_on_step", True)

    # @profile(stream=open('log_mem3_setup_fit2.log', 'w+'))
    # @profile
    def setup(self, stage="fit"):
        """Base LightningModule setup method. This will setup the datasets and dataloaders

        Note:
            We also send to the loggers all hyperparams that are not `nn.Module` (i.e. the policy).
            Apparently PyTorch Lightning does not do this by default.
        """
        log.info("Setting up batch sizes for train/val/test")
        train_bs, val_bs, test_bs = (
            self.data_cfg["batch_size"],
            self.data_cfg["val_batch_size"],
            self.data_cfg["test_batch_size"],
        )
        self.train_batch_size = train_bs
        self.val_batch_size = train_bs if val_bs is None else val_bs
        self.test_batch_size = self.val_batch_size if test_bs is None else test_bs

        log.info("Setting up datasets")

        # Create datasets automatically. If found, this will skip
        if self.data_cfg["generate_data"]:
            generate_default_datasets(data_dir=self.data_cfg["data_dir"], data_cfg=self.data_cfg)

        self.train_dataset = self.wrap_dataset(
            self.env.dataset(self.data_cfg["train_data_size"], phase="train")
        )
        self.val_dataset = self.env.dataset(self.data_cfg["val_data_size"], phase="val")
        self.test_dataset = self.env.dataset(
            self.data_cfg["test_data_size"], phase="test"
        )
        self.dataloader_names = None
        self.setup_loggers()
            
        self.post_setup_hook()

    def setup_loggers(self):
        """Log all hyperparameters except those in `nn.Module`"""
        if self.loggers is not None:
            hparams_save = {
                k: v for k, v in self.hparams.items() if not isinstance(v, nn.Module)
            }
            for logger in self.loggers:
                logger.log_hyperparams(hparams_save)
                logger.log_graph(self)
                logger.save()

    def post_setup_hook(self):
        """Hook to be called after setup. Can be used to set up subclasses without overriding `setup`"""
        pass

    def configure_optimizers(self, parameters=None):
        """
        Args:
            parameters: parameters to be optimized. If None, will use `self.policy.parameters()
        """

        # prog_optim_lst, prog_sche_dict = self.protagonist.configure_optimizers()
        prog_optim_dict = {}
        prog_optim = self.protagonist.configure_optimizers()
        if isinstance(prog_optim, tuple):
            prog_optim_cls, prog_lrsche_dict = prog_optim
            
            prog_optim_dict["optimizer"] = prog_optim_cls[0]
            prog_optim_dict["lr_scheduler"] = prog_lrsche_dict
        else:
            prog_optim_dict["optimizer"] = prog_optim_cls
            
        # adv_optim_lst, adv_sche_dict = self.adversary.configure_optimizers()
        adv_optim_dict = {}
        adv_optim = self.adversary.configure_optimizers()
        if isinstance(adv_optim, tuple):
            adv_optim_cls, adv_lrsche_dict = adv_optim
            adv_optim_dict["optimizer"] = adv_optim_cls[0]
            adv_optim_dict["lr_scheduler"] = adv_lrsche_dict
        else:
            adv_optim_dict["optimizer"] = adv_optim_cls
        
        # sche_dict = {"prog": prog_optim, "adv":adv_optim}
        return (prog_optim_dict, adv_optim_dict)

    def log_metrics(self, metric_dict: dict, phase: str, dataloader_idx: int = None):
        """Log metrics to logger and progress bar"""
        metrics = getattr(self, f"{phase}_metrics")
        dataloader_name = ""
        if dataloader_idx is not None and self.dataloader_names is not None:
            dataloader_name = "/" + self.dataloader_names[dataloader_idx]
        metrics = {
            f"{phase}/{k}{dataloader_name}": v.mean()
            if isinstance(v, torch.Tensor)
            else v
            for k, v in metric_dict.items()
            if k in metrics
        }
        log_on_step = self.log_on_step if phase == "train" else False
        # log_on_step = True
        # on_epoch = False if phase == "train" else True
        on_epoch = True
        self.log_dict(
            metrics,
            on_step=log_on_step,
            on_epoch=on_epoch,
            prog_bar=True,
            sync_dist=True,
            add_dataloader_idx=False,  # we add manually above
        )
        return metrics

    def forward(self, td, with_adv, phase, **kwargs):
        """Forward pass for the protagonist. ."""
        if kwargs.get("env", None) is None:
            env = self.env
        else:
            log.info("Using env from kwargs")
            env = kwargs.pop("env")
            
        if with_adv:    # adv disturb env
            out_adv = self.adversary(td.clone())
            td = env.reset_stochastic_demand(td, out_adv["action_adv"][..., None])    # env transition: get new real demand
        return self.protagonist(td, **kwargs)

    def shared_step(self, batch: Any, batch_idx: int, phase: str, **kwargs):
        """Shared step between train/val/test. To be implemented in subclass"""
        # phase = "train"
        # adv forward: change hyparams, env stochvar transition
        td, out_adv = self.adversary.inference_step(batch, batch_idx, phase)
        
        optim_prog, optim_adv = self.optimizers()
        
        loop_times = 1
        if phase == "train":
            loop_times = 1
            
        
        for _ in range(loop_times):
            # prog forward: get solution and update
            td_temp = td.clone()
            out_prog = self.protagonist.calculoss_step(td_temp, batch, phase)
            # if False:
            if phase == "train":
                prog_loss = out_prog["loss"]
                self.protagonist.man_update_step(prog_loss, optim_prog)
        
        td = td_temp.clone()
        out_all = {key: value for key, value in out_prog.items()}
        # ? change out_adv : reward? or td?
        # if False:
        if phase == "train":
            # adv update
            if self.current_epoch < 90:     # fix adv after 90 epoch
                if self.current_epoch % 5 == 0:
                    out_adv = self.adversary.update_step(td, out_adv, phase, optimizer=optim_adv)
        
                
        for key, value in out_adv.items():
            if "adv" not in key:
                key = "adv_" + key
            out_all[key] = value

        metrics = self.log_metrics(out_all, phase)
        # debug
        # if phase == "train":
        #     norms_prog_policy = grad_norm(self.protagonist.policy, norm_type=2)
        #     norms_adv_policy = grad_norm(self.adversary.policy, norm_type=2)
        #     norms_adv_critic = grad_norm(self.adversary.critic, norm_type=2)
            # print(f' prog grad norm: {norms_prog_policy["grad_2.0_norm_total"]} \
            #     adv, policy norm {norms_adv_policy["grad_2.0_norm_total"]}, \
            #     critic norm {norms_adv_critic["grad_2.0_norm_total"]}')
        return metrics
        # raise NotImplementedError("Shared step is required to implemented in subclass")

    def training_step(self, batch: Any, batch_idx: int):
        # To use new data every epoch, we need to call reload_dataloaders_every_epoch=True in Trainer
        
        return self.shared_step(batch, batch_idx, phase="train")

    def validation_step(self, batch: Any, batch_idx: int, dataloader_idx: int = None):
        return self.shared_step(
            batch, batch_idx, phase="val", dataloader_idx=dataloader_idx
        )

    def test_step(self, batch: Any, batch_idx: int, dataloader_idx: int = None):
        return self.shared_step(
            batch, batch_idx, phase="test", dataloader_idx=dataloader_idx
        )

    def train_dataloader(self):
        return self._dataloader(
            self.train_dataset, self.train_batch_size, self.shuffle_train_dataloader
        )

    def val_dataloader(self):
        return self._dataloader(self.val_dataset, self.val_batch_size)

    def test_dataloader(self):
        return self._dataloader(self.test_dataset, self.test_batch_size)

    def on_train_epoch_end(self):
        """Called at the end of the training epoch. This can be used for instance to update the train dataset
        with new data (which is the case in RL).
        """
        train_dataset = self.env.dataset(self.data_cfg["train_data_size"], "train")
        self.train_dataset = self.wrap_dataset(train_dataset)
        log.info("end of an epoch")
        print(f"end of an epoch, time {time.time()}")
        
        sche_prog, sche_adv = self.lr_schedulers()
        if isinstance(sche_prog, torch.optim.lr_scheduler.MultiStepLR):
            sche_prog.step()
        if isinstance(sche_adv, torch.optim.lr_scheduler.MultiStepLR):
            sche_adv.step()

    def wrap_dataset(self, dataset):
        """Wrap dataset with policy-specific wrapper. This is useful i.e. in REINFORCE where we need to
        collect the greedy rollout baseline outputs.
        """
        return dataset

    def _dataloader(self, dataset, batch_size, shuffle=False):
        """Handle both single datasets and list / dict of datasets"""
        if isinstance(dataset, Iterable):
            # load dataloader names if available as dict, else use indices
            if isinstance(dataset, dict):
                self.dataloader_names = list(dataset.keys())
            else:
                self.dataloader_names = [f"{i}" for i in range(len(dataset))]
            return [
                self._dataloader_single(ds, batch_size, shuffle)
                for ds in dataset.values()
            ]
        else:
            return self._dataloader_single(dataset, batch_size, shuffle)

    def _dataloader_single(self, dataset, batch_size, shuffle=False):
        """The dataloader used by the trainer. This is a wrapper around the dataset with a custom collate_fn
        to efficiently handle TensorDicts.
        """
        return DataLoader(
            dataset,
            batch_size=batch_size,
            shuffle=shuffle,
            num_workers=self.dataloader_num_workers,
            collate_fn=tensordict_collate_fn,
        )

def get_adversary_opponent(name, **kw):
    """Get a REINFORCE baseline by name
    The rollout baseline default to warmup baseline with one epoch of
    exponential baseline and the greedy rollout
    """

    opponent_cls = OPPONENTS_REGISTRY.get(name, None)
    if opponent_cls is None:
        raise ValueError(
            f"Unknown baseline {opponent_cls}. Available baselines: {OPPONENTS_REGISTRY.keys()}"
        )
    return opponent_cls

def get_adversary(name, **kw):
    """Get a adversary by name
    """

    adversary_cls = ADVERSARY_REGISTRY.get(name, None)
    if adversary_cls is None:
        raise ValueError(
            f"Unknown baseline {adversary_cls}. Available baselines: {ADVERSARY_REGISTRY.keys()}"
        )
    return adversary_cls

def get_protagonist(name, **kw):
    """Get a protagonist by name
    """

    protagonist_cls = PROTAGONIST_REGISTRY.get(name, None)
    if protagonist_cls is None:
        raise ValueError(
            f"Unknown baseline {protagonist_cls}. Available baselines: {PROTAGONIST_REGISTRY.keys()}"
        )
    return protagonist_cls
