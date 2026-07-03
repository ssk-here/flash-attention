"""Dump the optimized XLA HLO for jit-compiled naive attention and summarize
what the compiler fused — and, more importantly, what it could not fuse.

The point of this script is evidence. The README claims that XLA fuses the
elementwise softmax chain but cannot fuse *across* the two matmuls, so the
(N, N) score and probability matrices are materialized in HBM. Rather than
assert that, dump the compiler's own output and count:

  * `dot(` ops            — the two matmuls (QK^T and PV); fusion regions
                            stop at these boundaries.
  * `fusion(` computations — XLA's automatically fused elementwise kernels.
  * values typed f32[B,H,N,N] — every appearance is an N^2 buffer that some
                            kernel writes and another reads.
  * memory_analysis().temp_size_in_bytes — XLA's own accounting of scratch
                            memory. For naive attention it scales as N^2;
                            quadruple when N doubles.

Run at several sizes to see the scaling:

    python benchmarks/dump_hlo.py --seq-lens 256 512 1024

Full HLO text is written to results/hlo/ for reading alongside the summary.
Works on any backend (CPU included — the fusion-boundary structure is the
same; on GPU the dots additionally become cuBLAS custom-calls).
"""

import argparse
import re
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import jax
import jax.numpy as jnp

from flash_attention import naive_attention


def analyze(batch: int, heads: int, seq: int, head_dim: int, out_dir: Path):
    shape = (batch, heads, seq, head_dim)
    spec = jax.ShapeDtypeStruct(shape, jnp.float32)

    # lower() traces to StableHLO (what JAX hands to XLA); compile() runs the
    # full XLA optimization pipeline. The compiled text is where fusion
    # decisions are visible.
    lowered = jax.jit(naive_attention).lower(spec, spec, spec)
    compiled = lowered.compile()
    hlo = compiled.as_text()

    out_dir.mkdir(parents=True, exist_ok=True)
    path = out_dir / f"hlo_naive_seq{seq}.txt"
    path.write_text(hlo, encoding="utf-8")

    # --- count the structural features ------------------------------------
    # (Result types carry layout suffixes like f32[1,2,256,256]{3,2,1,0},
    # so match op names, not exact type strings.)
    n_dots = len(re.findall(r"\bdot\(", hlo))
    n_dots += len(re.findall(r"custom-call.*(cublas|gemm)", hlo))  # GPU form
    n_fusions = len(re.findall(r"\bfusion\(", hlo))
    # Every value whose *result type* ends in ...,N,N] is an N^2 tensor that
    # exists as a whole buffer between kernels.
    nn_results = len(re.findall(rf"= f32\[[0-9,]*{seq},{seq}\]", hlo))

    temp_bytes = None
    try:
        temp_bytes = compiled.memory_analysis().temp_size_in_bytes
    except Exception:
        pass  # memory_analysis is not implemented on every backend

    return dict(seq=seq, dots=n_dots, fusions=n_fusions, nn_buffers=nn_results,
                temp_bytes=temp_bytes, path=path)


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--seq-lens", type=int, nargs="+", default=[256, 512, 1024])
    p.add_argument("--batch", type=int, default=1)
    p.add_argument("--heads", type=int, default=2)
    p.add_argument("--head-dim", type=int, default=64)
    p.add_argument("--out-dir", type=Path, default=Path("results/hlo"))
    args = p.parse_args()

    print(f"backend={jax.default_backend()}  shape=(batch={args.batch}, "
          f"heads={args.heads}, N, d={args.head_dim})  dtype=f32\n")
    print(f"{'N':>6} {'dot ops':>8} {'fusions':>8} {'NxN vals':>9} "
          f"{'XLA temp memory':>16}   analytic 2*B*H*N^2*4")
    rows = []
    for n in args.seq_lens:
        r = analyze(args.batch, args.heads, n, args.head_dim, args.out_dir)
        rows.append(r)
        analytic = 2 * args.batch * args.heads * n * n * 4
        temp = f"{r['temp_bytes'] / 1e6:10.2f} MB" if r["temp_bytes"] else "     n/a"
        print(f"{n:>6} {r['dots']:>8} {r['fusions']:>8} {r['nn_buffers']:>9} "
              f"{temp:>16}   {analytic / 1e6:10.2f} MB")

    print("\nWhat to look for in the dumps (results/hlo/):")
    print(" * fusion(...) computations hold the entire scale->max->exp->sum")
    print("   ->divide softmax chain — XLA's automatic fusion working well.")
    print(" * but every fusion region STOPS at a dot( op: the f32[B,H,N,N]")
    print("   score and probability values are materialized between kernels.")
    print(" * temp memory therefore scales ~4x when N doubles. The Pallas")
    print("   kernel lowers to a single custom-call whose only large operands")
    print("   are the O(N*d) inputs/outputs — that gap IS FlashAttention.")
    for r in rows:
        print(f"   full HLO: {r['path']}")


if __name__ == "__main__":
    main()
