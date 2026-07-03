"""Benchmark the three attention implementations across sequence lengths.

For each sequence length N, the benchmark draws --repeats independent random
Q/K/V tensors (seeded: PRNGKey(seed), PRNGKey(seed+1), ...) so results are a
distribution, not a point estimate. Within a repeat, all implementations see
the same data. Writes one CSV row per (implementation, N, repeat) with median
wall time, achieved TFLOP/s, analytic HBM byte estimate and arithmetic
intensity.

Example (Colab L4 / TPU):
    python benchmarks/run_benchmark.py --seq-lens 512 1024 2048 4096 \
        --dtype bf16 --repeats 10 --output results/results.csv

On CPU-only hosts the Pallas kernel runs in interpret mode: correct but very
slow — keep --seq-lens small (e.g. 256 512) and treat timings as meaningless.
"""

import argparse
import csv
import statistics
import sys
import time
from collections import defaultdict
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import jax
import jax.numpy as jnp

from flash_attention import attention_xla, flash_attention, naive_attention

DTYPES = {"f32": jnp.float32, "bf16": jnp.bfloat16, "f16": jnp.float16}


def median_time_s(fn, args, warmup: int, iters: int) -> float:
    for _ in range(warmup):
        jax.block_until_ready(fn(*args))
    times = []
    for _ in range(iters):
        t0 = time.perf_counter()
        jax.block_until_ready(fn(*args))
        times.append(time.perf_counter() - t0)
    return statistics.median(times)


def attention_flops(b: int, h: int, n: int, d: int) -> float:
    """QKᵀ and PV: 2·N²·d MACs each → 4·N²·d FLOPs per head."""
    return 4.0 * b * h * n * n * d


def hbm_bytes(impl: str, b: int, h: int, n: int, d: int, elem: int) -> float:
    """Analytic HBM traffic estimate (lower bound, ignores caches).

    naive/xla: read Q,K,V + write O, plus the score/prob matrices — written
    once and read once each (XLA fuses scale+softmax so we charge 2 round
    trips, not 4; this flatters the baseline, which is fine).
    flash: only Q, K, V, O ever touch HBM.
    """
    qkvo = 4.0 * b * h * n * d * elem
    if impl == "pallas":
        return qkvo
    score_traffic = 4.0 * b * h * n * n * elem  # S write+read, P write+read
    return qkvo + score_traffic


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--seq-lens", type=int, nargs="+",
                   default=[512, 1024, 2048, 4096])
    p.add_argument("--batch", type=int, default=4)
    p.add_argument("--heads", type=int, default=8)
    p.add_argument("--head-dim", type=int, default=64)
    p.add_argument("--dtype", choices=DTYPES, default="f32")
    p.add_argument("--block-q", type=int, default=128)
    p.add_argument("--block-k", type=int, default=128)
    p.add_argument("--seed", type=int, default=0,
                   help="base PRNG seed; repeat r uses PRNGKey(seed + r)")
    p.add_argument("--repeats", type=int, default=10,
                   help="independent random data draws per sequence length")
    p.add_argument("--warmup", type=int, default=2)
    p.add_argument("--iters", type=int, default=5,
                   help="timed iterations per repeat (median taken)")
    p.add_argument("--impls", nargs="+", default=["naive", "xla", "pallas"],
                   choices=["naive", "xla", "pallas"])
    p.add_argument("--output", type=str, default="results/results.csv")
    args = p.parse_args()

    dtype = DTYPES[args.dtype]
    elem = jnp.dtype(dtype).itemsize
    backend = jax.default_backend()
    device = jax.devices()[0].device_kind
    print(f"backend={backend} device={device} dtype={args.dtype} "
          f"seed={args.seed} repeats={args.repeats}")
    if backend == "cpu":
        print("WARNING: CPU backend — Pallas runs in interpret mode; "
              "timings are for plumbing validation only.")

    # naive stays un-jitted on purpose (eager dispatch is what it measures).
    # The Pallas path is wrapped in a single jax.jit callable so re-tracing
    # of pallas_call is cached across calls and never counted in the timings.
    pallas_jit = jax.jit(lambda q, k, v: flash_attention(
        q, k, v, block_q=args.block_q, block_k=args.block_k))
    impls = {"naive": naive_attention, "xla": attention_xla, "pallas": pallas_jit}

    out_path = Path(args.output)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    rows = []
    for n in args.seq_lens:
        shape = (args.batch, args.heads, n, args.head_dim)
        flops = attention_flops(args.batch, args.heads, n, args.head_dim)
        times_by_impl = defaultdict(list)
        failed = set()

        for r in range(args.repeats):
            key = jax.random.PRNGKey(args.seed + r)
            kq, kk, kv = jax.random.split(key, 3)
            q = jax.random.normal(kq, shape, dtype)
            k = jax.random.normal(kk, shape, dtype)
            v = jax.random.normal(kv, shape, dtype)

            for name in args.impls:
                if name in failed:
                    continue
                try:
                    t = median_time_s(impls[name], (q, k, v),
                                      args.warmup, args.iters)
                except Exception as e:  # OOM on naive at long seqs is expected
                    print(f"  {name:6s} n={n:6d} r={r}  FAILED: "
                          f"{type(e).__name__}: {e}")
                    failed.add(name)
                    rows.append(dict(impl=name, seq_len=n, seed=args.seed + r,
                                     repeat=r, time_ms="", tflops_per_s="",
                                     est_hbm_gb="", arithmetic_intensity="",
                                     error=type(e).__name__))
                    continue
                times_by_impl[name].append(t)
                bytes_ = hbm_bytes(name, args.batch, args.heads, n,
                                   args.head_dim, elem)
                rows.append(dict(
                    impl=name, seq_len=n, seed=args.seed + r, repeat=r,
                    time_ms=f"{t * 1e3:.4f}",
                    tflops_per_s=f"{flops / t / 1e12:.4f}",
                    est_hbm_gb=f"{bytes_ / 1e9:.4f}",
                    arithmetic_intensity=f"{flops / bytes_:.2f}", error=""))

        for name in args.impls:
            ts = times_by_impl[name]
            if not ts:
                continue
            med = statistics.median(ts) * 1e3
            sd = statistics.stdev(ts) * 1e3 if len(ts) > 1 else 0.0
            print(f"  {name:6s} n={n:6d}  {med:10.3f} ± {sd:6.3f} ms  "
                  f"({len(ts)} repeats)  "
                  f"{flops / statistics.median(ts) / 1e12:8.2f} TFLOP/s")

    fieldnames = ["impl", "seq_len", "seed", "repeat", "time_ms",
                  "tflops_per_s", "est_hbm_gb", "arithmetic_intensity", "error"]
    meta = dict(batch=args.batch, heads=args.heads, head_dim=args.head_dim,
                dtype=args.dtype, backend=backend, device=device,
                seed=args.seed, repeats=args.repeats)
    with out_path.open("w", newline="") as f:
        f.write("# " + " ".join(f"{k}={v}" for k, v in meta.items()) + "\n")
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)
    print(f"wrote {out_path}")


if __name__ == "__main__":
    main()
