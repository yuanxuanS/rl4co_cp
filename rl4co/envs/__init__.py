# Base environment
from rl4co.envs.common.base import RL4COEnvBase

# for fix env: svrp_fix
from rl4co.envs.graph_pool import svrp_graph_pool
# EDA
from rl4co.envs.eda import DPPEnv, MDPPEnv

# Routing
from rl4co.envs.routing import (
    ATSPEnv,
    CVRPEnv,
    MTSPEnv,
    OPEnv,
    PCTSPEnv,
    PDPEnv,
    SDVRPEnv,
    SPCTSPEnv,
    TSPEnv,
    SVRPEnv,
    CSPEnv,
    SCPEnv,
)

# Scheduling
from rl4co.envs.scheduling import FFSPEnv, SMTWTPEnv

# Register environments
ENV_REGISTRY = {
    "atsp": ATSPEnv,
    "cvrp": CVRPEnv,
    "svrp": SVRPEnv,
    "dpp": DPPEnv,
    "ffsp": FFSPEnv,
    "mdpp": MDPPEnv,
    "mtsp": MTSPEnv,
    "op": OPEnv,
    "pctsp": PCTSPEnv,
    "pdp": PDPEnv,
    "sdvrp": SDVRPEnv,
    "spctsp": SPCTSPEnv,
    "tsp": TSPEnv,
    "csp": CSPEnv,
    "scp": SCPEnv,
    "smtwtp": SMTWTPEnv,
}


def get_env(env_name: str, *args, **kwargs) -> RL4COEnvBase:
    """Get environment by name.

    Args:
        env_name: Environment name
        *args: Positional arguments for environment
        **kwargs: Keyword arguments for environment

    Returns:
        Environment
    """
    env_cls = ENV_REGISTRY.get(env_name, None)
    if env_cls is None:
        raise ValueError(
            f"Unknown environment {env_name}. Available environments: {ENV_REGISTRY.keys()}"
        )
    return env_cls(*args, **kwargs)
