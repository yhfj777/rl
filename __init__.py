"""
PPO追踪模块
"""
from .core.agent import (
    Detection,
    Track,
    TrackingAgent,
    AgentManager
)
from .core.network import (
    PPONetwork,
    SharedPPOAgent,
    NetworkOutput
)
from .core.ppo import (
    PPOLearner,
    TrajectoryTracker,
    PPOTransition,
    TrajectoryInfo
)
from .core.environment import (
    TrackingEnvironment,
    EnvironmentState
)
from .data.dataset import (
    TrackingDataset,
    split_dataset
)
from .config.config import (
    PPOConfig,
    TrackingConfig,
    LoggingConfig
)

__all__ = [
    'Detection', 'Track', 'TrackingAgent', 'AgentManager',
    'PPONetwork', 'SharedPPOAgent', 'NetworkOutput',
    'PPOLearner', 'TrajectoryTracker', 'PPOTransition', 'TrajectoryInfo',
    'TrackingEnvironment', 'EnvironmentState',
    'TrackingDataset', 'split_dataset',
    'PPOConfig', 'TrackingConfig', 'LoggingConfig'
]

