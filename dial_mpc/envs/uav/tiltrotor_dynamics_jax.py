from dataclasses import dataclass
import functools
import jax
import jax.numpy as jnp
from dial_mpc.envs.uav.se3_dynamics_jax import SE3State, SE3Dynamics, SimpleSE3Dynamics, OveractuatedSE3Dynamics
import dial_mpc.envs.uav.se3_utils_jax as jse3
from dial_mpc.envs.uav.dynamics_jax import jitself, SingleIntegratorWithAngles

@functools.partial(jax.tree_util.register_dataclass,
                   data_fields=['g', 'Vb', 'alpha', 'alpha_dot'],
                   meta_fields=[])
@dataclass
class TiltrotorState(SE3State):
    """
    Class for a tiltrotor state in SE3
    Note that this allows for vectorized computations if g, Vb, alpha, alpha_dot are batched
    """

    alpha: jnp.ndarray
    alpha_dot: jnp.ndarray

    @property
    def SE3(self):
        return SE3State(self.g, self.Vb)
    
    def __hash__(self):
        return hash(type(self))
    
    def __eq__(self, other):
        return self is other

    @jitself
    def to_vector(self):
        parent_vector = super().to_vector()
        return jnp.concatenate([
            parent_vector,
            self.alpha,
            self.alpha_dot
        ], axis=-1)

    @classmethod
    def from_SE3alpha(cls, se3state: SE3State, alpha, alpha_dot):
        
        return cls(
                   g = se3state.g,
                   Vb = se3state.Vb,
                   alpha = alpha,
                   alpha_dot = alpha_dot
                   )

    @staticmethod
    def from_vector(x):
        se3vec = x[..., :18]
        alpha = x[..., 18:22]
        alpha_dot = x[..., 22:26]

        se3state = SE3State.from_vector(se3vec)
        return TiltrotorState.from_SE3alpha(se3state, alpha, alpha_dot)

    @classmethod
    def from_components(cls, p, R, v, omega, alpha, alpha_dot):
        se3state = super().from_components(p, R, v, omega)
        return TiltrotorState.from_SE3alpha(se3state, alpha, alpha_dot)

    @classmethod
    def identity(cls):
        return TiltrotorState.from_SE3alpha(super().identity(), jnp.zeros(4,), jnp.zeros(4,))

    @classmethod
    def zero_derivative(cls):
        return TiltrotorState.from_SE3alpha(super().zero_derivative(), jnp.zeros(4,), jnp.zeros(4,))

class TiltrotorDynamicsSimple(OveractuatedSE3Dynamics):
    """
    Simplified dynamics for a tiltrotor
    no multibody dynamics
    inputs are rotor thrust [0:4] and tilt speed [4:8]
    Ignores the alpha_dot component in TiltrotorState
    """

    def __init__(self, mass, inertia, 
                 max_u=1000, min_u=-1000, 
                 Ls=0.3*jnp.ones(4,), ctfs=8e-4*jnp.ones(4,), 
                 gravity=9.81, time_step=0.05) -> None:
        super().__init__(mass, inertia, max_u, min_u, Ls, ctfs, gravity, time_step)

        self.tilt_dynamics = SingleIntegratorWithAngles(4, jnp.inf, -jnp.inf, time_step)

    @jitself
    def __call__(self, state: TiltrotorState, u: jnp.ndarray, use_input_constraints=True) -> TiltrotorState:
        u = self.apply_input_constraints(u, use_input_constraints)

        force_vec = self.get_force_vec(state, u)
        se3_der = super().__call__(state, force_vec, False)
        alpha_dot = u[4:8]
        return TiltrotorState.from_SE3alpha(se3_der, alpha_dot, jnp.zeros(state.alpha_dot.shape))

    @jitself
    def get_force_vec(self, state: TiltrotorState, u: jnp.ndarray) -> jnp.ndarray:
        """ given u and state, return the lateral and vertical forces"""

        thrusts = u[..., :4]
        cos_comps = thrusts * jnp.cos(state.alpha)
        sin_comps = thrusts * jnp.sin(state.alpha)

        return jnp.stack((cos_comps, sin_comps), axis=-1).reshape(*thrusts.shape[:-1], -1)
    
    @jitself
    def integral_step(self, state: TiltrotorState, u: jnp.ndarray, dt=None) -> TiltrotorState:
        if dt is None:
            dt = self.time_step

        new_se3 = super().integral_step(state, u, dt)
        
        dstate = self.__call__(state, u)
        new_alpha = self.tilt_dynamics.integrate(state.alpha, dstate.alpha, total_time=dt)
        
        return TiltrotorState.from_SE3alpha(new_se3, new_alpha, state.alpha_dot)



class TiltrotorDynamicsFull(SimpleSE3Dynamics):
    """
    State is g, Vb, alpha, dalpha
    """
    pass