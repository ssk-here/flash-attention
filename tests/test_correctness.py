"""Correctness gate: the Pallas kernel must match the reference before any
benchmark number means anything. Runs on CPU via interpret mode."""

import jax
import jax.numpy as jnp
import pytest

from flash_attention import attention_xla, flash_attention, naive_attention


def _make_qkv(seq: int, dtype, batch: int = 2, heads: int = 4, head_dim: int = 64):
    kq, kk, kv = jax.random.split(jax.random.PRNGKey(0), 3)
    shape = (batch, heads, seq, head_dim)
    q = jax.random.normal(kq, shape, dtype)
    k = jax.random.normal(kk, shape, dtype)
    v = jax.random.normal(kv, shape, dtype)
    return q, k, v


@pytest.mark.parametrize("seq", [128, 256, 512])
def test_pallas_matches_naive_fp32(seq):
    q, k, v = _make_qkv(seq, jnp.float32)
    expected = naive_attention(q, k, v)
    actual = flash_attention(q, k, v)
    assert actual.shape == expected.shape
    assert jnp.allclose(actual, expected, atol=2e-3, rtol=2e-3), (
        f"max abs diff {jnp.max(jnp.abs(actual - expected)):.2e}"
    )


def test_pallas_matches_naive_bf16():
    q, k, v = _make_qkv(256, jnp.bfloat16)
    # Compare both in fp32 against an fp32 ground truth; bf16 has ~3 decimal
    # digits so tolerances are necessarily loose.
    expected = naive_attention(q.astype(jnp.float32), k.astype(jnp.float32),
                               v.astype(jnp.float32))
    actual = flash_attention(q, k, v).astype(jnp.float32)
    assert jnp.allclose(actual, expected, atol=2e-2, rtol=2e-2)


def test_xla_matches_naive():
    q, k, v = _make_qkv(256, jnp.float32)
    assert jnp.allclose(attention_xla(q, k, v), naive_attention(q, k, v),
                        atol=1e-5, rtol=1e-5)


def test_rejects_misaligned_seq():
    q, k, v = _make_qkv(128, jnp.float32)
    with pytest.raises(ValueError):
        flash_attention(q, k, v, block_q=96)


def test_non_square_seq():
    """seq_q != seq_k (e.g. cross-attention shapes)."""
    kq, kk, kv = jax.random.split(jax.random.PRNGKey(1), 3)
    q = jax.random.normal(kq, (1, 2, 256, 64), jnp.float32)
    k = jax.random.normal(kk, (1, 2, 512, 64), jnp.float32)
    v = jax.random.normal(kv, (1, 2, 512, 64), jnp.float32)
    expected = naive_attention(q, k, v)
    actual = flash_attention(q, k, v)
    assert jnp.allclose(actual, expected, atol=2e-3, rtol=2e-3)
