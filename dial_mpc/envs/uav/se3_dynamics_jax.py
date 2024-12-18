from dataclasses import dataclass
import functools
import jax
import jax.numpy as jnp
import dial_mpc.envs.uav.se3_utils_jax as jse3
from dial_mpc.envs.uav.dynamics_jax import Dynamics, jitself
# from jax import debug

@functools.partial(jax.tree_util.register_dataclass,
                   data_fields=['g', 'Vb'],
                   meta_fields=[])
@dataclass
class SE3State():
    """ 
    Class for a state in SE3 
    Note that this allows for vectorized computations if g and Vb are batched
    Also this class is indexable and iterable
    """

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
        return self.g[..., :3,3]
    
    @property
    def R(self):
        return self.g[..., :3,:3]
    
    @property
    def v(self):
        return self.Vb[..., :3]
    
    @property
    def omega(self):
        return self.Vb[..., 3:]
    
    @property
    def shape(self):
        return self.g.shape[:-2] # all descendents should include g
    
    @property
    def is_batched(self):
        return len(self.shape) > 0
    
    def __hash__(self):
        return hash(type(self))
    
    def __eq__(self, other):
        return self is other
    
    @jitself
    def __getitem__(self, idx):
        def get_idx(x):
            return x[idx] if len(self.shape) > 0 else x
        return jax.tree_util.tree_map(get_idx, self)
    
    @jitself
    def __len__(self):
        return self.shape[0] if self.is_batched else 1
    
    def __iter__(self):
        
        if self.is_batched:
            for i in range(len(self)):
                yield self[i]
        else:
            yield self

    @jitself
    def set(self, idx, value):
        """ 
        Returns a new datastructure with idx replaced by value
        (you can't actually mutate a jit-compilable structure)
        Note: does not work with boolean indices or slices
        """

        if not isinstance(value, self.__class__):
            raise TypeError(f"Can only be set with another {self.__class__.__name__}")
        
        return jax.tree_util.tree_map(
            lambda x, y: x.at[idx].set(y),
            self, value
        )
    
    @jitself
    def boolean_set(self, boolean, value):
        """ 
        Replaces self values with value where boolean is true
        Note: value should either be unbatched or the same size as self
        """
        
        if not isinstance(value, self.__class__):
            raise TypeError(f"Can only be set with another {self.__class__.__name__}")
        
        vec = self.to_vector()
        val_vec = value.to_vector()
        new_vec = jnp.where(boolean[..., None], val_vec, vec)
        
        return self.from_vector(new_vec)

    @jax.jit
    def append(state1, state2):
        """
        Returns a new datasstructure with state1 and state2 appended together
        (you can't actually mutate a jit-compatible data structure)
        Unbatched states are treated as having a shape of (1,)
        """

        if not isinstance(state2, state1.__class__):
            raise TypeError(f"Can only append with another {state1.__class__.__name__}")

        def maybe_add_batch_dim(x, is_batched):
            return x if is_batched else x[None, ...]
        
        arr1 = jax.tree_util.tree_map(
            lambda x: maybe_add_batch_dim(x, state1.is_batched),
            state1
        )

        arr2 = jax.tree_util.tree_map(
            lambda x: maybe_add_batch_dim(x, state2.is_batched),
            state2
        )

        return jax.tree_util.tree_map(
            lambda x, y: jnp.concatenate([x, y], axis=0),
            arr1, arr2
        )

    @jitself
    def to_vector(self):
        return jnp.concatenate([
            self.p,
            self.R.reshape((*self.R.shape[:-2], -1)),
            self.Vb
        ], axis=-1)
    
    def is_valid_rotation(self, tol=1e-6):
        #### NOT VECTORIZED YET
        R = self.R
        det_check = jnp.abs(jnp.linalg.det(R) - 1) < tol
        ortho_check = jnp.all(jnp.abs(R.T @ R - jnp.eye(3)) < tol)

        return det_check & ortho_check
    
    @staticmethod
    def from_pRVb(p, R, Vb):
        g = jnp.zeros((*p.shape[:-1], 4, 4))
        g = g.at[..., :3, :3].set(R)
        g = g.at[..., :3, 3].set(p)
        g = g.at[..., 3, 3].set(1.0)
        return SE3State(g, Vb)

    @staticmethod
    def from_components(p, R, v, omega):
        Vb = jnp.concatenate([v, omega], axis=-1)
        return SE3State.from_pRVb(p, R, Vb)
    
    @staticmethod
    def from_vector(x):
        p = x[..., :3]
        R = x[..., 3:12].reshape((*x.shape[:-1], 3,3))
        v = x[..., 12:15]
        omega = x[..., 15:18]
        return SE3State.from_components(p, R, v, omega)

    @staticmethod
    def identity():
        return SE3State(jnp.eye(4), jnp.zeros(6,))

    @staticmethod
    def zero_derivative():
        return SE3State(jnp.zeros((4,4)),jnp.zeros(6,))
    
class ThirdOrderSE3State(SE3State):
    """
    Class for a third order state in SE3
    """
    Vb_dot: jnp.ndarray

    @property
    def v_dot(self):
        return self.Vb_dot[..., :3]

    @property
    def omega_dot(self):
        return self.Vb_dot[..., 3:]
    
    def to_vector(self):
        parent_vector = super().to_vector()
        return jnp.concatenate([
            parent_vector, 
            self.Vb_dot
        ], axis=-1)

    @classmethod
    def from_SE3Vb_dot(cls, se3state: SE3State, Vb_dot: jnp.ndarray):
        return cls(
            g = se3state.g,
            Vb = se3state.Vb,
            Vb_dot = Vb_dot
        )
    
    ##### UNFINISHED

class SimpleSE3Dynamics(Dynamics):
    """ 
    Fully actuated second order simple mechanical system on SE3 
    - u is in R6 and represents body acceleration
    """

    def __init__(self, max_u = 1e3, min_u = -1e3, time_step = 0.05) -> None:

        xdim = 18
        udim = 6
        super().__init__(xdim, udim, max_u, min_u, time_step)

        self.e1 = jnp.array([1,0,0])
        self.e2 = jnp.array([0,1,0])
        self.e3 = jnp.array([0,0,1])

    @jitself
    def __call__(self, state: SE3State, u: jnp.ndarray, use_input_constraints=True) -> SE3State:
        """ 
        Returns a state object which actually contains the derivative (ie it's not actually a valid state)
        u is explicitly assumed to be a body acceleration

        Note! Make sure that u is batched to the same size as state
        """
        u = self.apply_input_constraints(u, use_input_constraints)
        
        omega_hat = jse3.skew3d(state.omega)        
        
        dp = self.batch_mat_vec_mul(state.R, state.v)
        dR = state.R @ omega_hat
        
        return SE3State.from_pRVb(dp, dR, u)
    
    @jitself
    def integral_step(self, state: SE3State, u: jnp.ndarray, dt=None) -> SE3State:
        """
        Explicitly integrates pose with fixed body velocity
        Single Euler step for higher orders
        """
        if dt is None:
            dt = self.time_step

        dstate = self.__call__(state, u)

        g2 = state.g @ jax.scipy.linalg.expm(jse3.hat3d(state.Vb)*dt)
        Vb2 = state.Vb + dstate.Vb*dt

        return SE3State(g2, Vb2)
    
    @jitself
    def integrate(self, state0: SE3State, u: jnp.ndarray, num_steps=10, total_time = None) -> SE3State:
        """
        total_time is total time for integration, split over num_steps
        """
        
        return super().integrate(state0, u, num_steps, total_time)
    
class SE3Dynamics(SimpleSE3Dynamics):
    """ Fully actuated second order mechanical system on SE3 """
    def __init__(self, mass, inertia,
                 max_u=1e3, min_u=-1e3,
                 gravity = 9.81, time_step=0.05) -> None:
        super().__init__(max_u, min_u, time_step)

        self.mass = mass
        if not inertia.shape == (3,3):
            raise TypeError('inertia must be a 3x3 matrix')
        try:
            chol = jnp.linalg.cholesky(inertia)
            if jnp.any(jnp.isnan(chol)):
                raise ValueError()
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

    @jitself
    def __call__(self, state: SE3State, u: jnp.ndarray, use_input_constraints=True) -> SE3State:
        """Note! make sure u is batched to the same dimension as state"""
        u = self.apply_input_constraints(u, use_input_constraints)
        
        drift, decoupling = self.get_drift_decoupling(state)

        dVb = drift + self.batch_mat_vec_mul(decoupling, u)

        return super().__call__(state, dVb, False)
    
    @jitself
    def get_gravity_wrench(self, state: SE3State) -> jnp.ndarray:
        g_gr = SE3State.from_pRVb(jnp.zeros(state.p.shape), state.R, state.Vb).g
        Fb_gravity = self.batch_mat_vec_mul(jse3.Adj_SE3(g_gr), self.gravity_wrench)
        return Fb_gravity

    @jitself
    def get_drift_decoupling(self, state: SE3State) -> tuple[jnp.ndarray, jnp.ndarray]:

        Fb_gravity = self.get_gravity_wrench(state)

        decoupling = self.inv_J

        adj_Vb_T = jnp.swapaxes(jse3.adj_se3(state.Vb), -2, -1)
        drift_ac = self.batch_mat_vec_mul(adj_Vb_T, self.batch_mat_vec_mul(self.J, state.Vb)) + Fb_gravity
        drift = self.batch_mat_vec_mul(self.inv_J, drift_ac)

        return drift, decoupling
    
    @jitself
    def get_Mf(self, state: SE3State) -> tuple[jnp.ndarray, jnp.ndarray]:

        Fb_gravity = self.get_gravity_wrench(state)

        M = self.J
        adj_Vb_T = jnp.swapaxes(jse3.adj_se3(state.Vb), -2, -1)
        f = - self.batch_mat_vec_mul(adj_Vb_T, self.batch_mat_vec_mul(self.J, state.Vb)) - Fb_gravity

        return M, f


class OveractuatedSE3Dynamics(SE3Dynamics):
    def __init__(self, mass, inertia, 
                 max_u=1000, min_u=-1000,
                 Ls = 0.3*jnp.ones(4,), ctfs = 8e-4*jnp.ones(4,), 
                 gravity=9.81, time_step=0.05) -> None:
        super().__init__(mass, inertia, max_u, min_u, gravity, time_step)

        self.udim = 8
        self.Ls = Ls
        self.ctfs = ctfs
        self.mixing_matrix = jnp.array([[0., 0., 0., 1., 0., 0., 0., -1.],
                                        [0., -1., 0., 0., 0., 1., 0., 0.],
                                        [1., 0., 1., 0., 1., 0., 1., 0.],
                                        [0., 0., Ls[1], -ctfs[1], 0., 0., -Ls[3], ctfs[3]],
                                        [-Ls[0], -ctfs[0], 0., 0., Ls[2], ctfs[2], 0., 0.],
                                        [ctfs[0], -Ls[0], -ctfs[1], -Ls[1], ctfs[2], -Ls[2], -ctfs[3], -Ls[3]]
                                        ])
        
        self.pinv_mixing_matrix = jnp.linalg.pinv(self.mixing_matrix)

    @jitself
    def get_drift_decoupling(self, state: SE3State) -> tuple[jnp.ndarray, jnp.ndarray]:

        drift, decoupling = super().get_drift_decoupling(state)
        decoupling = decoupling @ self.mixing_matrix
        return drift, decoupling
    
    @jitself
    def get_Mf(self, state: SE3State) -> tuple[jnp.ndarray, jnp.ndarray]:

        M, f = super().get_Mf(state)

        return self.pinv_mixing_matrix @ M, self.batch_mat_vec_mul(self.pinv_mixing_matrix, f)

def test_state():
    pass

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

def test_simple_dynamics_vectorized():
    ps = jnp.array([[0., 0., 0.],
                    [1., 0., 0.]])
    Rs = jnp.tile(jnp.eye(3)[None, ...], (2, 1, 1))
    vs = jnp.array([[1., 0., 0.],
                    [0., 1., 0.]])
    omegas = jnp.array([[0.0, 0.0, 0.0],
                        [0.1, 0.0, 0.0]])
    initial_states = SE3State.from_components(ps, Rs, vs, omegas)

    dynamics = SimpleSE3Dynamics()
    u = jnp.array([0.,0.,-9.81,0.,0.,0.])
    us = jnp.tile(u[None, ...], (2,1))

    final_states = dynamics.integrate(initial_states, us)
    print(final_states)

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

def test_full_dynamics_vectorized():
    ps = jnp.array([[0., 0., 0.],
                    [1., 0., 0.]])
    Rs = jnp.tile(jnp.eye(3)[None, ...], (2, 1, 1))
    vs = jnp.array([[1., 0., 0.],
                    [0., 1., 0.]])
    omegas = jnp.array([[0.0, 0.0, 0.0],
                        [0.1, 0.0, 0.0]])
    initial_states = SE3State.from_components(ps, Rs, vs, omegas)

    mass = 2
    inertia = jnp.diag(jnp.array([1.0, 1.2, 0.8]))

    dynamics = SE3Dynamics(mass, inertia)
    u = jnp.array([0.,0.,9.81,0.,0.,0.])
    us = jnp.tile(u[None, ...], (2,1))

    final_states = dynamics.integrate(initial_states, us)
    print(final_states)

def test_overactuated_dynamics():
    initial_state = SE3State.from_components(
        p=jnp.zeros(3),
        R=jnp.eye(3),
        v=jnp.array([1.,0.,0.]),
        omega=jnp.zeros(3)
    )

    mass = 2
    inertia = jnp.eye(3)


    dynamics = OveractuatedSE3Dynamics(mass, inertia)
    u = jnp.array([0.,0.,9.81,0.,0.,0., 0., 0.])

    final_state = dynamics.integrate(initial_state, u)
    print(final_state)

def test_overactuated_dynamics_vectorized():
    ps = jnp.array([[0., 0., 0.],
                    [1., 0., 0.]])
    Rs = jnp.tile(jnp.eye(3)[None, ...], (2, 1, 1))
    vs = jnp.array([[1., 0., 0.],
                    [0., 1., 0.]])
    omegas = jnp.array([[0.0, 0.0, 0.0],
                        [0.1, 0.0, 0.0]])
    initial_states = SE3State.from_components(ps, Rs, vs, omegas)

    mass = 2
    # inertia = jnp.diag(jnp.array([1.0, 1.2, 0.8]))
    inertia = jnp.eye(3)

    dynamics = OveractuatedSE3Dynamics(mass, inertia)
    u = jnp.array([0.,0.,9.81,0.,0.,0., 0., 0.])
    us = jnp.tile(u[None, ...], (2,1))

    final_states = dynamics.integrate(initial_states, us)
    print(final_states)

if __name__ == "__main__":
    test_simple_dynamics()
    test_simple_dynamics_vectorized()
    test_full_dynamics()
    test_full_dynamics_vectorized()
    test_overactuated_dynamics()
    test_overactuated_dynamics_vectorized()