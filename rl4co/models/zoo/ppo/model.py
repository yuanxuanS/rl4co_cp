from rl4co.envs import RL4COEnvBase
from rl4co.models.rl import PPO
from rl4co.models.rl.common.critic import CriticNetwork
from rl4co.models.zoo.ppo.policy import PPOPolicy


class PPOModel(PPO):
    """PPO Model based on Proximal Policy Optimization (PPO).

    Args:
        env: Environment to use for the algorithm
        policy: Policy to use for the algorithm
        critic: Critic to use for the algorithm
        policy_kwargs: Keyword arguments for policy
        critic_kwargs: Keyword arguments for critic
    """

    def __init__(
        self,
        env: RL4COEnvBase,
        policy: PPOPolicy = None,
        critic: CriticNetwork = None,
        policy_kwargs: dict = {},
        critic_kwargs: dict = {},
        **kwargs,
    ):
        if policy is None:
            policy = PPOPolicy(env.name, **policy_kwargs)

        if critic is None:
            critic = CriticNetwork(env.name, **critic_kwargs)

        super().__init__(env, policy, critic, **kwargs)
