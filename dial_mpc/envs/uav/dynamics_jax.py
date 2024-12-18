import jax
import jax.numpy as jnp

def jitself(fun):
    """ Decorator for jitting methods while preserving the self argument """
    return jax.jit(fun, static_argnums=0)

@jax.vmap
def wrap_angles(angs):
    return (angs + jnp.pi) % (2*jnp.pi) - jnp.pi

class Dynamics():
    def __init__(self, xdim, udim, max_u, min_u, time_step) -> None:
        self.xdim = xdim
        self.udim = udim
        self.max_u = max_u
        self.min_u = min_u
        self.time_step = time_step

    @jitself
    def batch_mat_vec_mul(self, A, b):
        """ Batch-compatible multiply for a matrix times a vector """

        return (A @ b[..., None])[..., 0]
    
    def apply_input_constraints(self, u, use_input_constraints=True):
        return jnp.where(use_input_constraints,
                         jnp.maximum(jnp.minimum(u, self.max_u), self.min_u),
                         u)
    
    def __call__(self, state: jnp.ndarray, u: jnp.ndarray, use_input_constraints=True) -> jnp.ndarray:
        raise NotImplementedError
    
    def integral_step(self, state0: jnp.ndarray, u: jnp.ndarray, dt=None) -> jnp.ndarray:
        raise NotImplementedError

    def integrate(self, state0: jnp.ndarray, u: jnp.ndarray, num_steps=None, total_time=None) -> jnp.ndarray:
        """ 
        total_time is total time for integration, split over num_steps
        """

        if total_time is None:
            total_time = self.time_step

        if num_steps is None:
            num_steps = 1

        dt = total_time / num_steps
        body_fun = lambda _, x: self.integral_step(x, u, dt)
        state = jax.lax.fori_loop(0, num_steps, body_fun, state0)

        return state

class LinearSystem(Dynamics):
    def __init__(self, 
                 A: jnp.ndarray, B: jnp.ndarray, C: jnp.ndarray, 
                 max_u = 1e3, min_u = 1e3, 
                 time_step = 0.05) -> None:
        
        xdim = A.shape[0]
        udim = B.shape[1]

        super().__init__(xdim, udim, max_u, min_u, time_step)


        self.A = A
        self.B = B
        self.C = C

        self.pinvA = jnp.linalg.pinv(A)

        self.max_u = max_u
        self.min_u = min_u

    @jitself
    def __call__(self, x: jnp.ndarray, u: jnp.ndarray, use_input_constraints=True) -> jnp.ndarray:
        u = self.apply_input_constraints(u, use_input_constraints)
        return self.batch_mat_vec_mul(self.A, x) + self.batch_mat_vec_mul(self.B, u)
    
    @jitself
    def integral_step(self, state0: jnp.ndarray, u: jnp.ndarray, dt=None) -> jnp.ndarray:
        """ 
        Explicitly integrate over dt with constant input u 
        if dt is None use self.time_step
        """

        if dt is None:
            dt = self.time_step

        u = self.apply_input_constraints(u, True)

        exp_AT = jax.scipy.linalg.expm(self.A * dt)
        int_exp_AT_B = self.pinvA @ ((exp_AT - jnp.eye(self.A.shape[0])) @ self.B)

        return self.batch_mat_vec_mul(exp_AT, state0) + self.batch_mat_vec_mul(int_exp_AT_B, u)
    
    def integrate(self, state0: jnp.ndarray, u: jnp.ndarray, num_steps=None, total_time=None) -> jnp.ndarray:
        """ 
        Explicitly integrate over time_step with constant input u 
        if total_time is None use self.time_step
        num_steps is not needed since we integrate explicitly, but for collision checking it can be used
        """

        return super().integrate(state0, u, num_steps, total_time)


class Integrator(LinearSystem):
    """
    Parent class for a bunch of integrators
    """

    def __init__(self, udim, degree, max_u=1000, min_u=1000, time_step=0.05) -> None:
        
        xdim = degree*udim
        
        A = jnp.eye(xdim, k=udim)
        B = jnp.zeros((xdim, udim))
        B = B.at[-udim:, :].set(jnp.eye(udim))
        C = jnp.zeros((udim, xdim))
        C = C.at[:, :udim].set(jnp.eye(udim))

        super().__init__(A, B, C, max_u, min_u, time_step)

        self.degree = degree

    @jitself
    def integral_step(self, state0: jnp.ndarray, u: jnp.ndarray, dt=None) -> jnp.ndarray:

        if dt is None:
            dt = self.time_step

        u = self.apply_input_constraints(u, True)

        Bd = self.B * dt
        power_of_A = jnp.eye(self.A.shape[0])
        factorial = 1
        for i in range(1, self.degree):
            power_of_A = power_of_A @ self.A
            factorial = factorial*(i+1)
            Bd = Bd + power_of_A @ self.B * (dt**(i+1)) / factorial

        Ad = jax.scipy.linalg.expm(self.A*dt)

        return self.batch_mat_vec_mul(Ad, state0) + self.batch_mat_vec_mul(Bd, u)

class SingleIntegrator(Integrator):
    def __init__(self, udim, max_u=1000, min_u=1000, time_step=0.05) -> None:
        super().__init__(udim, 1, max_u, min_u, time_step)

class DoubleIntegrator(Integrator):
    def __init__(self, udim, max_u=1000, min_u=1000, time_step=0.05) -> None:
        super().__init__(udim, 2, max_u, min_u, time_step)

class IntegratorWithAngles(Integrator):
    """Integrator, except the first state layer is an angle"""

    @jitself
    def wrap_angles(self, x):
        return x.at[..., :self.udim].set(wrap_angles(x[..., :self.udim]))

    @jitself
    def integrate(self, state0, u, num_steps=None, total_time=None):
        return self.wrap_angles(super().integrate(state0, u, num_steps, total_time))
    
class SingleIntegratorWithAngles(SingleIntegrator, IntegratorWithAngles):
    pass

class DoubleIntegratorWithAngles(DoubleIntegrator, IntegratorWithAngles):
    pass