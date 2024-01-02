from typing import Callable, Optional, Tuple, Union

import torch.nn as nn

from tensordict import TensorDict
from torch import Tensor
import torch
import torch.nn.functional as F
from rl4co.envs import RL4COEnvBase, get_env
from rl4co.models.nn.utils import get_log_likelihood
from rl4co.models.zoo.common.autoregressive.decoder import AutoregressiveDecoder
from rl4co.models.zoo.common.autoregressive.encoder import GraphAttentionEncoder
from rl4co.utils.pylogger import get_pylogger

log = get_pylogger(__name__)


class PPOContiPolicy(nn.Module):
    """Base Auto-regressive policy for NCO construction methods.
    The policy performs the following steps:
        1. Encode the environment initial state into node embeddings
        2. Decode (autoregressively) to construct the solution to the NCO problem
    Based on the policy from Kool et al. (2019) and extended for common use on multiple models in RL4CO.

    Note:
        We recommend to provide the decoding method as a keyword argument to the
        decoder during actual testing. The `{phase}_decode_type` arguments are only
        meant to be used during the main training loop. You may have a look at the
        evaluation scripts for examples.

    Args:
        env_name: Name of the environment used to initialize embeddings
        encoder: Encoder module. Can be passed by sub-classes.
        decoder: Decoder module. Can be passed by sub-classes.
        init_embedding: Model to use for the initial embedding. If None, use the default embedding for the environment
        context_embedding: Model to use for the context embedding. If None, use the default embedding for the environment
        dynamic_embedding: Model to use for the dynamic embedding. If None, use the default embedding for the environment
        embedding_dim: Dimension of the node embeddings
        num_encoder_layers: Number of layers in the encoder
        num_heads: Number of heads in the attention layers
        normalization: Normalization type in the attention layers
        mask_inner: Whether to mask the inner diagonal in the attention layers
        use_graph_context: Whether to use the initial graph context to modify the query
        sdpa_fn: Scaled dot product function to use for the attention
        train_decode_type: Type of decoding during training
        val_decode_type: Type of decoding during validation
        test_decode_type: Type of decoding during testing
        **unused_kw: Unused keyword arguments
        
        output 2 value: mu and std
    """

    def __init__(
        self,
        env_name: [str, RL4COEnvBase],
        policy_dis: str = "Beta",
        encoder: nn.Module = None,
        init_embedding: nn.Module = None,
        embedding_dim: int = 128,
        num_encoder_layers: int = 3,
        num_heads: int = 8,
        hidden_dim: int = 512,
        out_dim: int = 6,       # 2 weather(3 params)
        normalization: str = "batch",
        mask_inner: bool = True,
        use_graph_context: bool = True,
        sdpa_fn: Optional[Callable] = None,
        train_decode_type: str = "sampling",
        val_decode_type: str = "greedy",
        test_decode_type: str = "greedy",
        **unused_kw,
    ):
        super(PPOContiPolicy, self).__init__()

        if len(unused_kw) > 0:
            log.warn(f"Unused kwargs: {unused_kw}")

        if isinstance(env_name, RL4COEnvBase):
            env_name = env_name.name
        self.env_name = env_name

        if encoder is None:
            log.info("Initializing default GraphAttentionEncoder")
            self.encoder = GraphAttentionEncoder(
                env_name=self.env_name,
                num_heads=num_heads,
                embedding_dim=embedding_dim,
                num_layers=num_encoder_layers,
                normalization=normalization,
                init_embedding=init_embedding,
                sdpa_fn=sdpa_fn,
            )
        else:
            self.encoder = encoder

        self.conti_action_head = nn.Sequential(
            nn.Linear(embedding_dim, hidden_dim), nn.ReLU(), nn.Linear(hidden_dim, 6)
        )
        
        self.policy_dis = policy_dis
        self.train_decode_type = train_decode_type
        self.val_decode_type = val_decode_type
        self.test_decode_type = test_decode_type

    def forward(
        self,
        td: TensorDict,
        env: Union[str, RL4COEnvBase] = None,
        phase: str = "train",
        return_actions: bool = False,
        return_entropy: bool = True,
        return_init_embeds: bool = False,
    ) -> dict:
        """Forward pass of the policy.

        Args:
            td: TensorDict containing the environment state
            env: Environment to use for decoding
            phase: Phase of the algorithm (train, val, test)
            return_actions: Whether to return the actions
            return_entropy: Whether to return the entropy
            decoder_kwargs: Keyword arguments for the decoder

        Returns:
            out: Dictionary containing the reward, log likelihood, and optionally the actions and entropy
        """

        # ENCODER: get embeddings from initial state
        embeddings, init_embeds = self.encoder(td)

        actions = self.conti_action_head(embeddings).mean(1)    # [mu sigma]
        
        if self.policy_dis == "Beta":
            alpha = F.softplus(actions[:, :3]) + 1.
            beta = F.softplus(actions[:, 3:]) + 1.       # alpha and beta need to be larger than 1
            action_dist = torch.distributions.beta.Beta(alpha, beta)      # alpha, beta
            action_ = action_dist.sample()      # [batch, 3]
            log_prob_ = action_dist.log_prob(action_)
        
        if (phase == "train") | (phase == "test"):
            out = {
                "reward": td["reward"],
                "action_adv": action_,
                "log_likelihood_adv": log_prob_
            }
        elif phase == "val":
            out = {
                "action_adv": action_,
                "log_likelihood_adv": log_prob_
            }
        if return_entropy:
            entropy = -(log_prob_.exp() * log_prob_).nansum(dim=1)  # [batch]
            # entropy = entropy.sum(dim=1)  # [batch]
            out["entropy"] = entropy
            
            
        return out
        # # Instantiate environment if needed
        # if isinstance(env, str) or env is None:
        #     env_name = self.env_name if env is None else env
        #     log.info(f"Instantiated environment not provided; instantiating {env_name}")
        #     env = get_env(env_name)



        # Log likelihood is calculated within the model
        log_likelihood = get_log_likelihood(log_p, actions, td_out.get("mask", None))

        out = {
            "reward": td_out["reward"],
            "log_likelihood": log_likelihood,
        }
        if return_actions:
            out["actions"] = actions

        if return_entropy:
            entropy = -(log_p.exp() * log_p).nansum(dim=1)  # [batch, decoder steps]
            entropy = entropy.sum(dim=1)  # [batch]
            out["entropy"] = entropy

        if return_init_embeds:
            out["init_embeds"] = init_embeds

        return out

    def evaluate_action(
        self,
        td: TensorDict,
        action: Tensor,
        env: Union[str, RL4COEnvBase] = None,
    ) -> Tuple[Tensor, Tensor]:
        """Evaluate the action probability and entropy under the current policy

        Args:
            td: TensorDict containing the current state
            action: Action to evaluate
            env: Environment to evaluate the action in.
        """
    
        embeddings, _ = self.encoder(td)
        
        actions_curr = self.conti_action_head(embeddings).mean(1)    # [mu sigma]
        
        if self.policy_dis == "Beta":
            alpha = F.softplus(actions_curr[:, :3]) + 1.
            beta = F.softplus(actions_curr[:, 3:]) + 1.       # alpha and beta need to be larger than 1
            action_dist = torch.distributions.beta.Beta(alpha, beta)      # alpha, beta
            log_prob = action_dist.log_prob(action)      # action prob under current policy, [batch, 3]
            
            assert log_prob.isfinite().all(), "Log p is not finite"

            # compute entropy
            log_prob = torch.nan_to_num(log_prob, nan=0.0)
            entropy = -(log_prob.exp() * log_prob).sum(dim=-1)  # [batch, decoder steps]
            # entropy = entropy.sum(dim=1)  # [batch] -- sum over decoding steps
            assert entropy.isfinite().all(), "Entropy is not finite"
        return log_prob, entropy
