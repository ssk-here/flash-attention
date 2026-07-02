"""Fused FlashAttention forward kernel in JAX Pallas.

One program instance per (batch, head, q_block). The Q block stays resident
in fast on-chip memory (VMEM on TPU, SRAM on GPU) while K/V blocks stream
past it; the online-softmax recurrence tracks per-row running maxima and
normalizers so the (seq_q, seq_k) score matrix never exists in HBM.

The same kernel lowers through the Triton backend on GPU and the Mosaic
backend on TPU, and runs under interpret mode on CPU for correctness
checks. Running maxima/normalizers are kept as (block_q, 1) 2-D arrays —
TPU vector units want 2-D shapes.
"""

import functools
import math

import jax
import jax.numpy as jnp
from jax.experimental import pallas as pl


def _flash_attention_kernel(q_ref, k_ref, v_ref, o_ref, *, block_k: int, sm_scale: float):
    # q_ref: (block_q, d) — this program's resident Q block.
    # k_ref / v_ref: (seq_k, d) — full K/V for this (batch, head); streamed in tiles.
    q = q_ref[...]
    block_q, _ = q.shape
    seq_k = k_ref.shape[0]
    num_kv_blocks = seq_k // block_k

    def body(i, carry):
        acc, m_old, l_old = carry
        k = k_ref[pl.ds(i * block_k, block_k), :]
        v = v_ref[pl.ds(i * block_k, block_k), :]

        # S_block = Q Kᵀ, accumulated in fp32 regardless of input dtype.
        s = jax.lax.dot_general(
            q, k,
            dimension_numbers=(((1,), (1,)), ((), ())),
            preferred_element_type=jnp.float32,
        ) * sm_scale  # (block_q, block_k)

        m_new = jnp.maximum(m_old, jnp.max(s, axis=-1, keepdims=True))
        alpha = jnp.exp(m_old - m_new)          # rescale factor for history
        p = jnp.exp(s - m_new)                  # (block_q, block_k), fp32
        l_new = alpha * l_old + jnp.sum(p, axis=-1, keepdims=True)
        # P V_block on the matrix unit; cast P to the input dtype so bf16
        # inputs use the bf16 MXU/tensor-core path.
        pv = jax.lax.dot_general(
            p.astype(v.dtype), v,
            dimension_numbers=(((1,), (0,)), ((), ())),
            preferred_element_type=jnp.float32,
        )
        acc = acc * alpha + pv
        return acc, m_new, l_new

    acc0 = jnp.zeros((block_q, q_ref.shape[1]), dtype=jnp.float32)
    m0 = jnp.full((block_q, 1), -jnp.inf, dtype=jnp.float32)
    l0 = jnp.zeros((block_q, 1), dtype=jnp.float32)
    acc, _, l = jax.lax.fori_loop(0, num_kv_blocks, body, (acc0, m0, l0))

    o_ref[...] = (acc / l).astype(o_ref.dtype)


def flash_attention(
    q: jax.Array,
    k: jax.Array,
    v: jax.Array,
    *,
    block_q: int = 128,
    block_k: int = 128,
    interpret: bool | None = None,
) -> jax.Array:
    """FlashAttention forward pass (non-causal).

    Args:
        q, k, v: (batch, heads, seq, head_dim). seq_q must divide by block_q,
            seq_k by block_k. head_dim should be a multiple of 128 on TPU for
            best tiling; 64 also works.
        block_q, block_k: tile sizes. 128 aligns with TPU (8, 128) fp32 tiling
            and is a sane Triton default.
        interpret: force Pallas interpret mode. Defaults to True on CPU-only
            hosts so correctness tests run anywhere.

    Returns:
        (batch, heads, seq_q, head_dim), same dtype as q.
    """
    batch, heads, seq_q, head_dim = q.shape
    seq_k = k.shape[2]
    if seq_q % block_q != 0:
        raise ValueError(f"seq_q={seq_q} must be divisible by block_q={block_q}")
    if seq_k % block_k != 0:
        raise ValueError(f"seq_k={seq_k} must be divisible by block_k={block_k}")
    if interpret is None:
        interpret = jax.default_backend() == "cpu"

    sm_scale = 1.0 / math.sqrt(head_dim)
    grid = (batch, heads, seq_q // block_q)

    return pl.pallas_call(
        functools.partial(_flash_attention_kernel, block_k=block_k, sm_scale=sm_scale),
        grid=grid,
        in_specs=[
            # Q: one (block_q, d) tile per program.
            pl.BlockSpec((None, None, block_q, head_dim), lambda b, h, i: (b, h, i, 0)),
            # K, V: the full sequence for this (batch, head); the kernel
            # slices it into block_k tiles itself. Bounds VMEM use to
            # O(seq_k · d) per core — fine up to ~16k tokens at d=64 fp32.
            pl.BlockSpec((None, None, seq_k, head_dim), lambda b, h, i: (b, h, 0, 0)),
            pl.BlockSpec((None, None, seq_k, head_dim), lambda b, h, i: (b, h, 0, 0)),
        ],
        out_specs=pl.BlockSpec((None, None, block_q, head_dim), lambda b, h, i: (b, h, i, 0)),
        out_shape=jax.ShapeDtypeStruct((batch, heads, seq_q, head_dim), q.dtype),
        interpret=interpret,
    )(q, k, v)
