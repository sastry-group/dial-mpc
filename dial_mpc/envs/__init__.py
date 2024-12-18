from typing import Any, Dict, Sequence, Tuple, Union, List
from dial_mpc.envs.unitree_h1_env import (
    UnitreeH1WalkEnvConfig,
    UnitreeH1PushCrateEnvConfig,
    UnitreeH1LocoEnvConfig,
)
from dial_mpc.envs.unitree_go2_env import (
    UnitreeGo2EnvConfig,
    UnitreeGo2SeqJumpEnvConfig,
    UnitreeGo2CrateEnvConfig,
)

from dial_mpc.envs.uav.uav_env import UAVStationaryTrackingConfig, UAVStationaryTrackingEnv, TRTStationaryTrackingConfig, TRTStationaryTrackingEnv, TRTObstacleAvoidanceConfig, TRTObstacleAvoidanceEnv

_configs = {
    "unitree_h1_walk": UnitreeH1WalkEnvConfig,
    "unitree_h1_push_crate": UnitreeH1PushCrateEnvConfig,
    "unitree_h1_loco": UnitreeH1LocoEnvConfig,
    "unitree_go2_walk": UnitreeGo2EnvConfig,
    "unitree_go2_seq_jump": UnitreeGo2SeqJumpEnvConfig,
    "unitree_go2_crate_climb": UnitreeGo2CrateEnvConfig,
    "uav_stationary_tracking": UAVStationaryTrackingConfig,
    "trt_stationary_tracking": TRTStationaryTrackingConfig,
    "trt_obstacle_avoidance": TRTObstacleAvoidanceConfig,
}

_custom_envs = {
    "uav_stationary_tracking": UAVStationaryTrackingEnv,
    "trt_stationary_tracking": TRTStationaryTrackingEnv,
    "trt_obstacle_avoidance": TRTObstacleAvoidanceEnv,
}

def register_config(name: str, config: Any):
    _configs[name] = config


def get_config(name: str) -> Any:
    return _configs[name]

def register_custom_env(name: str, env: Any):
    _custom_envs[name] = env

def get_custom_env(name: str) -> Any:
    return _custom_envs[name]
