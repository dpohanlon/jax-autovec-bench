import numpy as np

import tensorflow as tf

import jax.numpy as jnp
from jax import jit, vmap

from pprint import pprint

import time

@jit
def filter_vec(p_proj, C_proj, H, G, m):

    HG = H.T @ G

    # Innermost two axes must be 'matrix'
    inv_C_proj = jnp.linalg.inv(C_proj)

    C = jnp.linalg.inv(inv_C_proj + HG @ H)

    p = jnp.einsum('Bij,Bj->Bi', inv_C_proj, p_proj) + jnp.einsum('ji,iB->Bj', HG, m)
    p = jnp.einsum('Bij,Bj->Bi', C, p)

    return p, C

@jit
def filter_novec(p_proj, C_proj, H, G, m):

    inv_C_proj = jnp.linalg.inv(C_proj)
    HG = H.T @ G

    C = jnp.linalg.inv(inv_C_proj + HG @ H)

    p = inv_C_proj.dot(p_proj) + HG.dot(m)
    p = C.dot(p)

    return p, C

@tf.function
def filter_tf(p_proj, C_proj, H, G, m):

    HG = tf.transpose(H) @ G

    # Innermost two axies must be 'matrix'
    inv_C_proj = tf.linalg.inv(C_proj)

    C = tf.linalg.inv(inv_C_proj + HG @ H)

    # Reversing batch dimension -> fix me!
    p = tf.einsum('Bij,Bj->Bi', inv_C_proj, p_proj) + tf.einsum('ji,iB->Bj', HG, m)
    p = tf.einsum('Bij,Bj->Bi', C, p)

    return p, C

if __name__ == '__main__':

    n_batch = 1024

    pB = np.random.normal(size = (n_batch, 4))
    CB = np.random.normal(size = (n_batch, 4, 4))

    p = pB[0]
    C = CB[0]

    H = np.ones((4, 4))
    G = np.ones((4, 4))

    mB = np.random.normal(size = (n_batch, 4)).T
    m = mB[:, 0]

    p_f_tf, C_f_tf = filter_tf(pB, CB, H, G, mB)
    p_f_v, C_f_v = filter_vec(pB, CB, H, G, mB)

    p_f_nv, C_f_nv = filter_novec(p, C, H, G, m)

    pprint(p_f_tf)
    pprint(p_f_v)
    pprint(p_f_nv)

    filter_vmap = jit(vmap(filter_novec, in_axes=(0, 0, None, None, 1)))

    p_f_av, C_f_av = filter_vmap(pB, CB, H, G, mB)

    pprint(p_f_av)
