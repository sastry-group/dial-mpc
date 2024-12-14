from dataclasses import dataclass

import jax
import jax.numpy as jnp
from functools import partial
from jax.scipy.spatial.transform import Rotation as R

from uav_plan.dynamics.se3_dynamics_jax import SE3State

@dataclass
class StationaryTrackingRewardConfig():
    target_position: jnp.ndarray
    target_orientation: jnp.ndarray
    weight_position: float = 1.0
    weight_orientation: float = 1.0
    weight_velocity: float = 1.0
    weight_angular_velocity: float = 1.0
    weight_action: float = 1.0

class StationaryTrackingReward():
    def __init__(self, config: StationaryTrackingRewardConfig):
        """
        Args:
            target_position: (3,) x, y, z position of the target (meter)
            target_orientation: (3,) roll, pitch, yaw of the target (radian)
        """
        self.target_position = config.target_position
        self.target_orientation = R.from_euler('xyz', config.target_orientation, degrees=False).as_matrix()
        self.weight_position = config.weight_position
        self.weight_orientation = config.weight_orientation
        self.weight_velocity = config.weight_velocity
        self.weight_angular_velocity = config.weight_angular_velocity
        self.weight_action = config.weight_action

    @partial(jax.jit, static_argnums=0)
    def __call__(self, state: jnp.ndarray, action: jnp.ndarray) -> jnp.ndarray:
        assert (len(state.shape) == 2 and state.shape[-1] == 18) or (len(state.shape) == 1 and state.shape[-1] == 18), "State must be a 2D array with shape (N, 18) or a 1D array with shape (18,)"
        p = state[..., :3]
        rot = state[..., 3:12].reshape(state.shape[:-1] + (3, 3))
        v = state[..., 12:15]
        omega = state[..., 15:]

        position_reward = -1.0 * self.weight_position * jnp.linalg.norm(p - self.target_position, axis=-1)
        orientation_reward = -1.0 * self.weight_orientation * 0.5 * jnp.trace(jnp.eye(3) - rot.swapaxes(-1, -2) @ self.target_orientation, axis1=-2, axis2=-1)
        velocity_reward = -1.0 * self.weight_velocity * jnp.linalg.norm(v, axis=-1)
        angular_velocity_reward = -1.0 * self.weight_angular_velocity * jnp.linalg.norm(omega, axis=-1)
        action_reward = -1.0 * self.weight_action * jnp.linalg.norm(action, axis=-1)
        return position_reward + orientation_reward + velocity_reward + angular_velocity_reward + action_reward

def test_stationary_tracking_reward():
    config = StationaryTrackingRewardConfig(
        target_position=jnp.array([0.0, 0.0, 0.0]),
        target_orientation=jnp.array([0.0, 0.0, 3.14]),
    )
    reward = StationaryTrackingReward(config)
    state = jnp.zeros((18,))
    state = state.at[3:12].set(R.from_euler('xyz', [0.0, 0.0, 0.0], degrees=False).as_matrix().flatten())
    state = state.at[12:15].set(jnp.array([0.0, 0.0, 0.0]))

    action = jnp.zeros((6,))
    print(reward(state, action))

    # test multiple states
    state = jnp.repeat(state[None, ...], 10, axis=0)
    action = jnp.repeat(action[None, ...], 10, axis=0)
    print(reward(state, action))

if __name__ == "__main__":
    test_stationary_tracking_reward()