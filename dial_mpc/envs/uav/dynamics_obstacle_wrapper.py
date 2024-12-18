from dataclasses import dataclass
import functools
import jax
import jax.numpy as jnp
from dial_mpc.envs.uav.dynamics_jax import Dynamics, jitself
from dial_mpc.envs.uav.se3_dynamics_jax import SE3State, SimpleSE3Dynamics
from abc import ABC, abstractmethod
from typing import List


def make_test_env(mass: float=2.0, inertia: jnp.ndarray=jnp.eye(3), time_step: float=0.01):
    from dial_mpc.envs.uav.tiltrotor_dynamics_jax import TiltrotorState, TiltrotorDynamicsSimple

    Ls = 0.3 * jnp.ones(4,)
    ctfs = 8e-4 * jnp.ones(4,)
    max_u = 15.0
    min_u = -max_u

    dynamics = TiltrotorDynamicsSimple(mass, inertia, max_u, min_u, Ls, ctfs, time_step=time_step)

    initial_state = TiltrotorState(
        g = jnp.eye(4),
        Vb = jnp.zeros(6,),
        alpha = jnp.zeros(4,),
        alpha_dot = jnp.zeros(4,)
        )

    goal_state = TiltrotorState.from_components(
        p = jnp.array([5,0,0]),
        R = jnp.eye(3),
        v = jnp.zeros(3,),
        omega = jnp.zeros(3,),
        alpha = jnp.zeros(4,),
        alpha_dot = jnp.zeros(4,)
        ) # Note we're only using p right now
    

    box = Box3d(max_corner = jnp.array([3, 2, 2]), min_corner = jnp.array([2, -2, -2]))

    obstacle_cost = 1000
    input_cost = jnp.array([10, 10, 10, 10, 5, 5, 5, 5])
    distance_cost = 10

    wrapper = DynamicsObstacleWrapper(
        dynamics= dynamics,
        obstacles= [box],
        obstacle_cost= obstacle_cost,
        input_cost= input_cost,
        distance_cost= distance_cost
    )

    return wrapper



class Obstacle(ABC):
    """Abstract base class for an obstacle in R^N space"""
    @abstractmethod
    def check_collision(self, position: jnp.ndarray) -> jnp.ndarray:
        """
        Check if input position(s) lie within the obstacle
        Args:
            position: (..., N) where N is the degree of the space
        Returns:
            (...) boolean array indicated collition status
        """
        pass

    # def __hash__(self):
    #     return hash(type(self))
    
    # def __eq__(self, other):
    #     return self is other

    @abstractmethod
    def get_cost(self, position: jnp.ndarray, inside_cost: float, steepness: float, outside_cost: float) -> jnp.ndarray:
        """
        Returns obstacle cost for each input position(s)
        Args:
            position: (..., N) where N is the degree of the space
        Returns:
            (...) float array indicating cost
        """
        
        pass

    @staticmethod
    def sigmoid(signed_distance: jnp.ndarray, steepness: float) -> jnp.ndarray:
        return 1.0 / (1.0 + jnp.exp(signed_distance * steepness))
    
    @staticmethod
    def sigmoid_cost(signed_distance: jnp.ndarray, inside_cost: float, steepness: float, outside_cost: float) -> jnp.ndarray:
        return outside_cost + (inside_cost - outside_cost) * Obstacle.sigmoid(signed_distance, steepness)

@functools.partial(jax.tree_util.register_dataclass,
                   data_fields=['center', 'radius'],
                   meta_fields=[])
@dataclass
class Sphere(Obstacle):
    """ Sphere obstacle in 3D space """
    center: jnp.ndarray # (3,) array
    radius: float

    def __hash__(self):
        return hash(type(self))
    
    def __eq__(self, other):
        return self is other

    @jitself
    def check_collision(self, position: jnp.ndarray) -> jnp.ndarray:
        diff = position - self.center
        dist_sq = jnp.sum(diff*diff, axis=-1)
        return dist_sq <= self.radius**2
    
    @jitself
    def get_cost(self, position: jnp.ndarray, inside_cost: float = 1000.0, steepness: float = 10.0, outside_cost: float = 0.0) -> jnp.ndarray:
        diff = position - self.center
        dist = jnp.sqrt(jnp.sum(diff*diff, axis=-1))
        signed_dist = dist - self.radius
        return self.sigmoid_cost(signed_dist, inside_cost, steepness, outside_cost)



@functools.partial(jax.tree_util.register_dataclass,
                   data_fields=['min_corner', 'max_corner'],
                   meta_fields=[])
@dataclass
class Box3d(Obstacle):
    """ Axis-aligned box obstacle in 3D space """
    min_corner: jnp.ndarray # (3,) array
    max_corner: jnp.ndarray # (3,) array

    def __hash__(self):
        return hash(type(self))
    
    def __eq__(self, other):
        return self is other

    @jitself
    def check_collision(self, position: jnp.ndarray) -> jnp.ndarray:
        above_min = jnp.all(position >= self.min_corner, axis=-1)
        below_max = jnp.all(position <= self.max_corner, axis=-1)
        return above_min & below_max
    
    @jitself
    def get_cost(self, position: jnp.ndarray, inside_cost: float = 1000.0, steepness: float = 10.0, outside_cost: float = 0.0) -> jnp.ndarray:
        center = (self.max_corner + self.min_corner)/2.0
        half_size = (self.max_corner - self.min_corner)/2.0

        dist_from_center_per_dim = jnp.abs(position - center)
        signed_dists_per_dim = dist_from_center_per_dim - half_size

        max_signed_dist = jnp.max(signed_dists_per_dim, axis=-1)
        return self.sigmoid_cost(max_signed_dist, inside_cost, steepness, outside_cost)


class DynamicsObstacleWrapper(Dynamics):
    """
    Parent class for an obstacle wrapper
    Contains a collection of obstacles, an "in-collision" method
    Modifies the dynamics to stay stationary if an integral step is in collision
    """

    def __init__(self, dynamics: SimpleSE3Dynamics, obstacles: List[Obstacle],
                 obstacle_cost: float, input_cost: jnp.ndarray, distance_cost: float) -> None:
        super().__init__(dynamics.xdim, dynamics.udim, dynamics.max_u, dynamics.min_u, dynamics.time_step)
        self.dynamics = dynamics
        self.obstacles = obstacles
        self.obstacle_cost = obstacle_cost
        if input_cost is None:
            input_cost = jnp.ones(self.udim)
        self.input_cost = input_cost
        self.distance_cost = distance_cost

    @jitself
    def check_collision(self, state) -> jnp.ndarray:
        
        position = state.p
        collision = jnp.zeros(position.shape[:-1], dtype=bool)

        for obstacle in self.obstacles:
            collision = collision | obstacle.check_collision(position)

        return collision
    
    @jitself
    def get_obstacle_cost(self, state: SE3State) -> jnp.ndarray:
        position = state.p
        overall_cost = jnp.zeros(position.shape[:-1])

        for obstacle in self.obstacles:
            obstacle_cost = obstacle.get_cost(position, self.obstacle_cost, self.obstacle_cost / 10.0, 0.0)
            overall_cost = jnp.maximum(overall_cost, obstacle_cost)

        return overall_cost
    
    @jitself
    def get_goal_cost(self, state: SE3State, goal: SE3State) -> jnp.ndarray:
        diff = goal.p - state.p
        dist = jnp.sum(diff * diff, axis=-1)
        return self.distance_cost * dist
    
    @jitself
    def get_total_cost(self, state: SE3State, u: jnp.ndarray, goal: SE3State) -> jnp.ndarray:
        input_cost = u @ self.input_cost
        return input_cost + self.get_goal_cost(state, goal) + self.get_obstacle_cost(state) 
    
    @jitself
    def __call__(self, state: SE3State, u: jnp.ndarray, use_input_constraints=True) -> SE3State:
        """ Freezes dynamics when in collision """
        dstate = self.dynamics.__call__(state, u, use_input_constraints)
        collision = self.check_collision(state)

        return dstate.boolean_set(collision, state.zero_derivative())

    @jitself
    def integral_step(self, state0: SE3State, u: jnp.ndarray, dt=None) -> SE3State:
        new_state = self.dynamics.integral_step(state0, u, dt)
        collision = self.check_collision(state0)

        return new_state.boolean_set(collision, state0)