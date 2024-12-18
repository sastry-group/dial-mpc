from dataclasses import dataclass
from typing import Any, Dict, Sequence, Tuple, Union, List

import numpy as np

import jax
import jax.numpy as jnp
from functools import partial
from jax.scipy.spatial.transform import Rotation as R

from brax.envs.base import State
from brax.base import State as BraxState

from dial_mpc.envs.uav.stationary_tracking import StationaryTrackingReward, StationaryTrackingRewardConfig
from dial_mpc.envs.uav.se3_dynamics_jax import SE3State, SE3Dynamics
from dial_mpc.envs.uav.tiltrotor_dynamics_jax import TiltrotorState, TiltrotorDynamicsSimple
from dial_mpc.envs.uav.dynamics_obstacle_wrapper import DynamicsObstacleWrapper, make_test_env

@dataclass
class UAVStationaryTrackingConfig:
    task_name: str = "default"
    dt: float = 0.02
    timestep: float = 0.002
    backend: str = "custom"
    action_scale: float = 1.0
    target_position: jnp.ndarray = jnp.array([0.0, 0.0, 0.0])
    target_orientation: jnp.ndarray = jnp.array([0.0, 0.0, 0.0])
    weight_position: float = 1.0
    weight_orientation: float = 1.0
    weight_velocity: float = 1.0
    weight_angular_velocity: float = 1.0
    weight_action: float = 1.0

class UAVStationaryTrackingEnv:
    def __init__(self, config: UAVStationaryTrackingConfig):
        self.config = config
        self.backend = config.backend
        reward_config = StationaryTrackingRewardConfig(
            target_position=config.target_position,
            target_orientation=config.target_orientation,
            weight_position=config.weight_position,
            weight_orientation=config.weight_orientation,
            weight_velocity=config.weight_velocity,
            weight_angular_velocity=config.weight_angular_velocity,
            weight_action=config.weight_action,
        )
        self.reward = StationaryTrackingReward(reward_config)

        mass = 2
        inertia = jnp.eye(3)
        self.dynamics = SE3Dynamics(mass, inertia, time_step=config.timestep)
        self.initial_state = SE3State.from_components(
            p=jnp.zeros(3),
            R=jnp.eye(3),
            v=jnp.array([0.,0.,0.]),
            omega=jnp.zeros(3)
        )

        self.decimation_factor = int(config.dt / config.timestep)
        assert config.dt % config.timestep == 0, "Timestep must be divisible by dt"

        self.action_size = 6
        self.nx = 18
        self.nu = 6

    def reset(self, rng: jax.Array):
        pipeline_state = BraxState(
            q=jnp.concatenate([self.initial_state.p, self.initial_state.R.flatten()]),
            qd=jnp.concatenate([self.initial_state.v, self.initial_state.omega]),
            x=jnp.zeros(3),
            xd=jnp.zeros(3),
            contact=None,
        )
        reward = self.reward(self.initial_state.to_vector(), jnp.zeros(6))
        obs, done = jnp.zeros(2)
        metrics = {}
        state_info = {
            "rng": rng,
            "pos_tar": self.config.target_position,
            "vel_tar": jnp.zeros(3),
            "ang_vel_tar": jnp.zeros(3),
            "orientation_tar": self.config.target_orientation,
            "step": 0,
            "se3_state": self.initial_state,
        }
        state = State(pipeline_state, obs, reward, done, metrics, state_info)
        return state

    def step(self, state: State, action: jax.Array):
        # step dynamics
        action_scaled = action * self.config.action_scale
        next_state = self.dynamics.integrate(state.info["se3_state"], action_scaled, num_steps=self.decimation_factor)
        reward = self.reward(next_state.to_vector(), action)
        state.info["step"] += 1
        state.info["se3_state"] = next_state
        pipeline_state = BraxState(
            q=jnp.concatenate([next_state.p, next_state.R.flatten()]),
            qd=jnp.concatenate([next_state.v, next_state.omega]),
            x=jnp.zeros(3),
            xd=jnp.zeros(3),
            contact=None,
        )
        state = state.replace(
            pipeline_state=pipeline_state, reward=reward
        )
        return state

@dataclass
class TRTStationaryTrackingConfig:
    task_name: str = "default"
    dt: float = 0.02
    timestep: float = 0.002
    backend: str = "custom"
    action_scale: float = 1.0
    target_position: jnp.ndarray = jnp.array([0.0, 0.0, 0.0])
    target_orientation: jnp.ndarray = jnp.array([0.0, 0.0, 0.0])
    weight_position: float = 1.0
    weight_orientation: float = 1.0
    weight_velocity: float = 1.0
    weight_angular_velocity: float = 1.0
    weight_action: float = 1.0

class TRTStationaryTrackingEnv:
    def __init__(self, config: TRTStationaryTrackingConfig):
        self.config = config
        self.backend = config.backend
        reward_config = StationaryTrackingRewardConfig(
            target_position=config.target_position,
            target_orientation=config.target_orientation,
            weight_position=config.weight_position,
            weight_orientation=config.weight_orientation,
            weight_velocity=config.weight_velocity,
            weight_angular_velocity=config.weight_angular_velocity,
            weight_action=config.weight_action,
        )
        self.reward = StationaryTrackingReward(reward_config)

        mass = 2
        inertia = jnp.eye(3)
        self.dynamics = TiltrotorDynamicsSimple(mass, inertia, time_step=config.timestep)
        self.initial_state = TiltrotorState.from_components(
            p=jnp.zeros(3),
            R=jnp.eye(3),
            v=jnp.array([0.,0.,0.]),
            omega=jnp.zeros(3),
            alpha=jnp.zeros(4),
            alpha_dot=jnp.zeros(4)
        )

        self.decimation_factor = int(config.dt / config.timestep)
        assert config.dt % config.timestep == 0, "Timestep must be divisible by dt"

        self.action_size = 8
        self.nx = 26
        self.nu = 8

    def reset(self, rng: jax.Array):
        pipeline_state = BraxState(
            q=jnp.concatenate([self.initial_state.p, self.initial_state.R.flatten(), self.initial_state.alpha]),
            qd=jnp.concatenate([self.initial_state.v, self.initial_state.omega, self.initial_state.alpha_dot]),
            x=jnp.zeros(3),
            xd=jnp.zeros(3),
            contact=None,
        )
        reward = self.reward(self.initial_state.to_vector(), jnp.zeros(8))
        obs, done = jnp.zeros(2)
        metrics = {}
        state_info = {
            "rng": rng,
            "pos_tar": self.config.target_position,
            "vel_tar": jnp.zeros(3),
            "ang_vel_tar": jnp.zeros(3),
            "orientation_tar": self.config.target_orientation,
            "step": 0,
            "se3_state": self.initial_state,
        }
        state = State(pipeline_state, obs, reward, done, metrics, state_info)
        return state

    def step(self, state: State, action: jax.Array):
        # step dynamics
        action_scaled = action * self.config.action_scale
        next_state = self.dynamics.integrate(state.info["se3_state"], action_scaled, num_steps=self.decimation_factor)
        reward = self.reward(next_state.to_vector(), action)
        state.info["step"] += 1
        state.info["se3_state"] = next_state
        pipeline_state = BraxState(
            q=jnp.concatenate([next_state.p, next_state.R.flatten(), next_state.alpha]),
            qd=jnp.concatenate([next_state.v, next_state.omega, next_state.alpha_dot]),
            x=jnp.zeros(3),
            xd=jnp.zeros(3),
            contact=None,
        )
        state = state.replace(
            pipeline_state=pipeline_state, reward=reward
        )
        return state

@dataclass
class TRTObstacleAvoidanceConfig:
    task_name: str = "default"
    dt: float = 0.02
    timestep: float = 0.002
    backend: str = "custom"
    action_scale: float = 1.0
    target_position: jnp.ndarray = jnp.array([0.0, 0.0, 0.0])
    target_orientation: jnp.ndarray = jnp.array([0.0, 0.0, 0.0])
    weight_position: float = 1.0
    weight_orientation: float = 1.0
    weight_velocity: float = 1.0
    weight_angular_velocity: float = 1.0
    weight_action: float = 1.0

class TRTObstacleAvoidanceEnv:
    def __init__(self, config: TRTObstacleAvoidanceConfig):
        self.config = config
        self.backend = config.backend
        reward_config = StationaryTrackingRewardConfig(
            target_position=config.target_position,
            target_orientation=config.target_orientation,
            weight_position=config.weight_position,
            weight_orientation=config.weight_orientation,
            weight_velocity=config.weight_velocity,
            weight_angular_velocity=config.weight_angular_velocity,
            weight_action=config.weight_action,
        )

        self.state_input_reward = StationaryTrackingReward(reward_config)

        mass = 2
        inertia = jnp.eye(3)
        self.dynamics = make_test_env(mass, inertia, config.timestep)
        self.initial_state = TiltrotorState.from_components(
            p=jnp.zeros(3),
            R=jnp.eye(3),
            v=jnp.array([0.,0.,0.]),
            omega=jnp.zeros(3),
            alpha=jnp.zeros(4),
            alpha_dot=jnp.zeros(4)
        )

        self.decimation_factor = int(config.dt / config.timestep)
        assert config.dt % config.timestep == 0, "Timestep must be divisible by dt"

        self.action_size = 8
        self.nx = 26
        self.nu = 8

    def reset(self, rng: jax.Array):
        pipeline_state = BraxState(
            q=jnp.concatenate([self.initial_state.p, self.initial_state.R.flatten(), self.initial_state.alpha]),
            qd=jnp.concatenate([self.initial_state.v, self.initial_state.omega, self.initial_state.alpha_dot]),
            x=jnp.zeros(3),
            xd=jnp.zeros(3),
            contact=None,
        )
        reward = self.state_input_reward(self.initial_state.to_vector(), jnp.zeros(8)) + self.dynamics.get_obstacle_cost(self.initial_state)
        obs, done = jnp.zeros(2)
        metrics = {}
        state_info = {
            "rng": rng,
            "pos_tar": self.config.target_position,
            "vel_tar": jnp.zeros(3),
            "ang_vel_tar": jnp.zeros(3),
            "orientation_tar": self.config.target_orientation,
            "step": 0,
            "se3_state": self.initial_state,
        }
        state = State(pipeline_state, obs, reward, done, metrics, state_info)
        return state

    def step(self, state: State, action: jax.Array):
        # step dynamics
        action_scaled = action * self.config.action_scale
        next_state = self.dynamics.integrate(state.info["se3_state"], action_scaled, num_steps=self.decimation_factor)
        reward = self.state_input_reward(next_state.to_vector(), action) - self.dynamics.get_obstacle_cost(next_state)
        # jax.debug.print("reward: {reward}", reward=reward)
        state.info["step"] += 1
        state.info["se3_state"] = next_state
        pipeline_state = BraxState(
            q=jnp.concatenate([next_state.p, next_state.R.flatten(), next_state.alpha]),
            qd=jnp.concatenate([next_state.v, next_state.omega, next_state.alpha_dot]),
            x=jnp.zeros(3),
            xd=jnp.zeros(3),
            contact=None,
        )
        state = state.replace(
            pipeline_state=pipeline_state, reward=reward
        )
        return state