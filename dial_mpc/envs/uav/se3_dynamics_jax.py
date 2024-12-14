from dataclasses import dataclass
import functools
import jax
import jax.numpy as jnp
import uav_plan.utils.se3_utils_jax as jse3
# from jax import debug

@functools.partial(jax.tree_util.register_dataclass,
                   data_fields=['g', 'Vb'],
                   meta_fields=[])
@dataclass
class SE3State():
    """ Class for a state in SE3 """

    g: jnp.ndarray
    Vb: jnp.ndarray

    # Actually I don't want debug rotation matrix checking so I can store dx here too
    # def __post_init__(self):
    #     R = self.R
    #     det_check = jnp.abs(jnp.linalg.det(R) - 1) < 1e-6
    #     ortho_check = jnp.all(jnp.abs(R.T @ R - jnp.eye(3)) < 1e6)
    #     debug.check(det_check, ortho_check, "Invalid Rotation Matrix")

    @property
    def p(self):
        return self.g[:3,3]
    
    @property
    def R(self):
        return self.g[:3,:3]
    
    @property
    def v(self):
        return self.Vb[:3]
    
    @property
    def omega(self):
        return self.Vb[3:]
    
    def to_vector(self):
        return jnp.concatenate([
            self.p,
            self.R.reshape(-1),
            self.Vb
        ])
    
    def is_valid_rotation(self, tol=1e-6):
        R = self.R
        det_check = jnp.abs(jnp.linalg.det(R) - 1) < tol
        ortho_check = jnp.all(jnp.abs(R.T @ R - jnp.eye(3)) < tol)

        return det_check & ortho_check
    
    @classmethod
    def from_pRVb(cls, p, R, Vb):
        g = jnp.zeros((4,4))
        g = g.at[:3, :3].set(R)
        g = g.at[:3, 3].set(p)
        g = g.at[3,3].set(1.0)
        return cls(g, Vb)

    @classmethod
    def from_components(cls, p, R, v, omega):
        Vb = jnp.concatenate([v, omega])
        return cls.from_pRVb(p, R, Vb)
    
    @classmethod
    def from_vector(cls, x):
        p = x[:3]
        R = x[3:12].reshape(3,3)
        v = x[12:15]
        omega = x[15:18]
        return cls.from_components(p, R, v, omega)

class SimpleSE3Dynamics():
    """ Simple mechanical system on SE3 """

    def __init__(self, max_u = 1e3, min_u = -1e3, time_step = 0.05) -> None:
        self.max_u = max_u
        self.min_u = min_u
        self.time_step = time_step

        self.e1 = jnp.array([1,0,0])
        self.e2 = jnp.array([0,1,0])
        self.e3 = jnp.array([0,0,1])

    @staticmethod
    def jitself(fun):
        return jax.jit(fun, static_argnums=0)

    @jitself
    def __call__(self, state: SE3State, u: jnp.ndarray, input_constraints=True) -> SE3State:

        u = jnp.where(input_constraints,
                      jnp.maximum(jnp.minimum(u, self.max_u), self.min_u),
                      u)
        
        omega_hat = jse3.skew3d(state.omega)        
        dp = state.R @ state.v
        dR = state.R @ omega_hat
        
        return SE3State.from_pRVb(dp, dR, u)
    
    @jitself
    def integral_step(self, state: SE3State, u: jnp.ndarray, dt=None) -> SE3State:

        if dt is None:
            dt = self.time_step

        dstate = self.__call__(state, u)

        g2 = state.g @ jax.scipy.linalg.expm(jse3.hat3d(state.Vb)*dt)
        Vb2 = state.Vb + dstate.Vb*dt

        return SE3State(g2, Vb2)
    
    @jitself
    def integrate(self, state0: SE3State, u: jnp.ndarray, num_steps=10, time_step = None) -> SE3State:
        if time_step is None:
            time_step = self.time_step

        dt = time_step/num_steps
        # state = state0

        body_fun = lambda _, x: self.integral_step(x, u, dt)
        state = jax.lax.fori_loop(0, num_steps, body_fun, state0)
        # Jax way of doing
        # for _ in range(num_steps): state = self.integral_step(state, u, dt)

        return state
    
class SE3Dynamics(SimpleSE3Dynamics):
    def __init__(self, mass, inertia,
                 max_u=1e3, min_u=-1e3,
                 gravity = 9.81, time_step=0.05) -> None:
        super().__init__(max_u, min_u, time_step)

        self.mass = mass
        if not inertia.shape == (3,3):
            raise TypeError('inertia must be a 3x3 matrix')
        try:
            jnp.linalg.cholesky(inertia)
        except:
            raise ValueError('inertia must be positive definite')
        
        self.inertia = inertia
        self.inv_inertia = jnp.linalg.inv(self.inertia)

        J = jnp.zeros((6,6))
        J = J.at[:3,:3].set(self.mass*jnp.eye(3))
        self.J = J.at[3:,3:].set(self.inertia)
        self.inv_J = jnp.linalg.inv(self.J)

        self.gravity = gravity
        self.gravity_wrench = jnp.array([0., 0., -self.mass*self.gravity, 0.,0.,0.])

    @staticmethod
    def jitself(fun):
        return jax.jit(fun, static_argnums=0)

    @jitself
    def __call__(self, state: SE3State, u: jnp.ndarray, input_constraints=True) -> SE3State:
        u = jnp.where(input_constraints,
                      jnp.maximum(jnp.minimum(u, self.max_u), self.min_u),
                      u)
        
        drift, decoupling = self.get_drift_decoupling(state)

        dVb = drift + decoupling @ u

        return super().__call__(state, dVb, False)

    @jitself
    def get_drift_decoupling(self, state: SE3State) -> tuple[jnp.ndarray, jnp.ndarray]:

        g_gr = SE3State.from_pRVb(jnp.zeros(3), state.R, state.Vb).g
        Fb_gravity = jse3.Adj_SE3(g_gr) @ self.gravity_wrench

        decoupling = self.inv_J
        drift_ac = jse3.adj_se3(state.Vb).T @ self.J @ state.Vb + Fb_gravity
        drift = self.inv_J @ drift_ac

        return drift, decoupling









def test_simple_dynamics():
    initial_state = SE3State.from_components(
        p=jnp.zeros(3),
        R=jnp.eye(3),
        v=jnp.array([1.,0.,0.]),
        omega=jnp.zeros(3)
    )

    dynamics = SimpleSE3Dynamics()
    u = jnp.array([0.,0.,-9.81,0.,0.,0.])

    final_state = dynamics.integrate(initial_state, u)
    print(final_state)

def test_full_dynamics():
    initial_state = SE3State.from_components(
        p=jnp.zeros(3),
        R=jnp.eye(3),
        v=jnp.array([1.,0.,0.]),
        omega=jnp.zeros(3)
    )

    mass = 2
    inertia = jnp.eye(3)

    dynamics = SE3Dynamics(mass, inertia)
    u = jnp.array([0.,0.,9.81,0.,0.,0.])

    final_state = dynamics.integrate(initial_state, u)
    print(final_state)


if __name__ == "__main__":
    test_simple_dynamics()
    test_full_dynamics()