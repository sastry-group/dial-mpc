import jax
import jax.numpy as jnp

@jax.jit
def skew3d(v: jnp.ndarray) -> jnp.ndarray:

    result = jnp.zeros((*v.shape[:-1], 3, 3))
    v0, v1, v2 = v[..., 0], v[..., 1], v[..., 2]

    result = result.at[..., 0, 1].set(-v2)
    result = result.at[..., 0, 2].set(v1)
    result = result.at[..., 1, 0].set(v2)
    result = result.at[..., 1, 2].set(-v0)
    result = result.at[..., 2, 0].set(-v1)
    result = result.at[..., 2, 1].set(v0)
    
    return result

@jax.jit
def hat3d(Vb: jnp.ndarray) -> jnp.ndarray:
    v = Vb[..., :3]
    omega = Vb[..., 3:]
    omega_hat = skew3d(omega)

    xi_hat = jnp.zeros((*Vb.shape[:-1], 4, 4))

    xi_hat = xi_hat.at[..., :3, :3].set(omega_hat)
    xi_hat = xi_hat.at[..., :3, 3].set(v)
    
    return xi_hat

@jax.jit
def Adj_SE3(g: jnp.ndarray) -> jnp.ndarray:
    R = g[..., :3,:3]
    p = g[..., :3, 3]

    Adj = jnp.zeros((*g.shape[:-2], 6, 6))
    Adj = Adj.at[..., :3,:3].set(R)
    Adj = Adj.at[..., :3,3:].set(skew3d(p) @ R)
    # Adj = Adj.at[3:,:3].set(jnp.zeros((3,3)))
    Adj = Adj.at[..., 3:,3:].set(R)

    return Adj

@jax.jit
def adj_se3(Vb: jnp.ndarray) -> jnp.ndarray:
    v = Vb[..., :3]
    w = Vb[..., 3:]
    v_skew = skew3d(v)
    w_skew = skew3d(w)

    adj = jnp.zeros((*Vb.shape[:-1], 6,6))
    adj = adj.at[..., :3,:3].set(w_skew)
    adj = adj.at[..., :3,3:].set(v_skew)
    # adj = adj.at[3:,:3].set(jnp.zeros((3,3)))
    adj = adj.at[..., 3:,3:].set(w_skew)

    return adj