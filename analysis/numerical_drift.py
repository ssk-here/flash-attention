"""Numerical drift of the online-softmax recurrence vs. sequence length.

The online softmax used by FlashAttention is *algebraically* exact: in real
arithmetic it computes the same result as plain softmax. In floating point
it does extra work — every K/V block applies a rescale exp(m_old - m_new)
to the running accumulator — so it has its own, different rounding-error
profile. Correctness tests check one size at one tolerance; this experiment
measures how the error of each implementation *scales* with N, against a
float64 ground truth:

    variants:
      naive  f32   — plain softmax attention, all-f32
      pallas f32   — tiled online-softmax kernel, all-f32
      naive  bf16  — plain attention executed in bf16 end to end
      pallas bf16  — bf16 inputs/outputs, but fp32 accumulators inside the
                     kernel (preferred_element_type) — the mixed-precision
                     design point real kernels use

Questions the plot answers:
  1. Does the online recurrence drift faster than plain softmax as N grows?
     (If the fp32 pallas and fp32 naive curves track each other, no.)
  2. How much accuracy do fp32 in-kernel accumulators buy back for bf16
     inputs? (Distance between the two bf16 curves.)

Error metrics are computed against the f64 reference: max elementwise |err|
and relative Frobenius error. Multiple seeded draws give error bars.

Runs anywhere (CPU uses Pallas interpret mode — numerics, not speed, are
what's measured, so interpret results are meaningful here).

    python analysis/numerical_drift.py --seq-lens 128 256 512 1024 2048 4096
"""

import argparse
import csv
import statistics
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import jax

# Must happen before any array is created: enables float64 for the reference.
jax.config.update("jax_enable_x64", True)

import jax.numpy as jnp  # noqa: E402
import matplotlib  # noqa: E402

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

from flash_attention import flash_attention, naive_attention  # noqa: E402


def measure(n: int, seed: int, batch: int, heads: int, head_dim: int):
    """One data draw at size n -> {variant: (max_abs, rel_fro)}."""
    kq, kk, kv = jax.random.split(jax.random.PRNGKey(seed), 3)
    shape = (batch, heads, n, head_dim)
    q64 = jax.random.normal(kq, shape, jnp.float64)
    k64 = jax.random.normal(kk, shape, jnp.float64)
    v64 = jax.random.normal(kv, shape, jnp.float64)

    # Ground truth: everything in float64. ~15 decimal digits dwarfs any
    # f32/bf16 effect being measured.
    ref = naive_attention(q64, k64, v64)
    ref_norm = jnp.sqrt(jnp.sum(ref * ref))

    def err(out):
        d = out.astype(jnp.float64) - ref
        return (float(jnp.max(jnp.abs(d))),
                float(jnp.sqrt(jnp.sum(d * d)) / ref_norm))

    results = {}
    for dtype, tag in ((jnp.float32, "f32"), (jnp.bfloat16, "bf16")):
        q, k, v = q64.astype(dtype), k64.astype(dtype), v64.astype(dtype)
        results[f"naive_{tag}"] = err(naive_attention(q, k, v))
        results[f"pallas_{tag}"] = err(flash_attention(q, k, v))
    return results


VARIANTS = ["naive_f32", "pallas_f32", "naive_bf16", "pallas_bf16"]
STYLE = {
    "naive_f32":   ("tab:orange", "s", "-"),
    "pallas_f32":  ("tab:blue",   "^", "-"),
    "naive_bf16":  ("tab:orange", "s", "--"),
    "pallas_bf16": ("tab:blue",   "^", "--"),
}


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--seq-lens", type=int, nargs="+",
                   default=[128, 256, 512, 1024, 2048, 4096])
    p.add_argument("--batch", type=int, default=1)
    p.add_argument("--heads", type=int, default=2)
    p.add_argument("--head-dim", type=int, default=64)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--repeats", type=int, default=3)
    p.add_argument("--out-dir", type=Path, default=Path("results/numerical-drift"))
    args = p.parse_args()

    args.out_dir.mkdir(parents=True, exist_ok=True)
    print(f"backend={jax.default_backend()}  reference=float64  "
          f"repeats={args.repeats}")

    rows = []  # (variant, n, repeat, max_abs, rel_fro)
    for n in args.seq_lens:
        for r in range(args.repeats):
            res = measure(n, args.seed + r, args.batch, args.heads, args.head_dim)
            for variant, (max_abs, rel_fro) in res.items():
                rows.append((variant, n, r, max_abs, rel_fro))
        summary = "  ".join(
            f"{v}={statistics.mean(x[3] for x in rows if x[0] == v and x[1] == n):.2e}"
            for v in VARIANTS)
        print(f"  N={n:>5}  max|err| vs f64:  {summary}")

    csv_path = args.out_dir / "numerical_drift.csv"
    with csv_path.open("w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["variant", "seq_len", "repeat", "max_abs_err", "rel_fro_err"])
        w.writerows(rows)
    print(f"wrote {csv_path}")

    # ---- plot: max|err| and relative Frobenius error vs N -----------------
    fig, axes = plt.subplots(1, 2, figsize=(11, 4.5))
    for ax, (idx, ylabel) in zip(axes, [(3, "max elementwise |error|"),
                                        (4, "relative Frobenius error")]):
        for v in VARIANTS:
            color, marker, ls = STYLE[v]
            xs, ys, es = [], [], []
            for n in args.seq_lens:
                vals = [row[idx] for row in rows if row[0] == v and row[1] == n]
                xs.append(n)
                ys.append(statistics.mean(vals))
                es.append(statistics.stdev(vals) if len(vals) > 1 else 0.0)
            ax.errorbar(xs, ys, yerr=es, color=color, marker=marker,
                        linestyle=ls, capsize=3, label=v)
        ax.set_xscale("log", base=2)
        ax.set_yscale("log")
        ax.set_xlabel("sequence length N")
        ax.set_ylabel(ylabel)
        ax.grid(True, which="both", alpha=0.3)
        ax.legend(fontsize=8)
    fig.suptitle("Attention output error vs. float64 reference "
                 "(solid = f32 inputs, dashed = bf16 inputs)", fontsize=10)
    fig.tight_layout()
    out = args.out_dir / "drift_vs_seqlen.png"
    fig.savefig(out, dpi=150)
    print(f"wrote {out}")


if __name__ == "__main__":
    main()
