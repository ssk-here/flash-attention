"""Reference attention: materializes the full (seq_q, seq_k) score matrix.

Run un-jitted this is the eager baseline; wrapped in jax.jit it becomes the
XLA baseline (see __init__.py). Same math either way, which is the point:
the only variable across the three implementations is how the computation
is scheduled onto the hardware.
"""

import math

import jax
import jax.numpy as jnp


def naive_attention(q: jax.Array, k: jax.Array, v: jax.Array) -> jax.Array:
    """Standard scaled dot-product attention.

    Args:
        q: (batch, heads, seq_q, head_dim)
        k: (batch, heads, seq_k, head_dim)
        v: (batch, heads, seq_k, head_dim)

    Returns:
        (batch, heads, seq_q, head_dim)
    """
    scale = 1.0 / math.sqrt(q.shape[-1])
    scores = jnp.einsum("bhqd,bhkd->bhqk", q, k) * scale  # (B, H, N, N) hits HBM
    probs = jax.nn.softmax(scores, axis=-1)               # and again
    return jnp.einsum("bhqk,bhkd->bhqd", probs, v)
