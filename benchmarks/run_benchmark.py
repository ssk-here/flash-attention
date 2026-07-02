"""Benchmark the three attention implementations across sequence lengths.

Writes a CSV with median wall time, achieved TFLOP/s, analytic HBM byte
estimates and arithmetic intensity per (implementation, seq_len) pair.

Example (Colab L4 / TPU):
    python benchmarks/run_benchmark.py --seq-lens 512 1024 2048 4096 \
        --dtype bf16 --output results/results.csv

On CPU-only hosts the Pallas kernel runs in interpret mode: correct but very
slow — keep --seq-lens small (e.g. 256 512) and treat timings as meaningless.
"""

import argparse
import csv
import statistics
import sys
import time
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
    p.add_argument("--warmup", type=int, default=3)
    p.add_argument("--iters", type=int, default=10)
    p.add_argument("--impls", nargs="+", default=["naive", "xla", "pallas"],
                   choices=["naive", "xla", "pallas"])
    p.add_argument("--output", type=str, default="results/results.csv")
    args = p.parse_args()

    dtype = DTYPES[args.dtype]
    elem = jnp.dtype(dtype).itemsize
    backend = jax.default_backend()
    device = jax.devices()[0].device_kind
    print(f"backend={backend} device={device} dtype={args.dtype}")
    if backend == "cpu":
        print("WARNING: CPU backend — Pallas runs in interpret mode; "
              "timings are for plumbing validation only.")

    impls = {
        "naive": naive_attention,
        "xla": attention_xla,
        "pallas": lambda q, k, v: flash_attention(
            q, k, v, block_q=args.block_q, block_k=args.block_k),
    }

    out_path = Path(args.output)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    rows = []
    for n in args.seq_lens:
        shape = (args.batch, args.heads, n, args.head_dim)
        kq, kk, kv = jax.random.split(jax.random.PRNGKey(0), 3)
        q = jax.random.normal(kq, shape, dtype)
        k = jax.random.normal(kk, shape, dtype)
        v = jax.random.normal(kv, shape, dtype)
        flops = attention_flops(args.batch, args.heads, n, args.head_dim)

        for name in args.impls:
            try:
                t = median_time_s(impls[name], (q, k, v), args.warmup, args.iters)
            except Exception as e:  # OOM on the naive path at long seqs is expected
                print(f"  {name:6s} n={n:6d}  FAILED: {type(e).__name__}: {e}")
                rows.append(dict(impl=name, seq_len=n, time_ms="", tflops_per_s="",
                                 est_hbm_gb="", arithmetic_intensity="",
                                 error=type(e).__name__))
                continue
            bytes_ = hbm_bytes(name, args.batch, args.heads, n, args.head_dim, elem)
            tflops = flops / t / 1e12
            intensity = flops / bytes_
            print(f"  {name:6s} n={n:6d}  {t * 1e3:10.3f} ms  "
                  f"{tflops:8.2f} TFLOP/s  AI={intensity:8.1f}")
            rows.append(dict(impl=name, seq_len=n, time_ms=f"{t * 1e3:.4f}",
                             tflops_per_s=f"{tflops:.4f}",
                             est_hbm_gb=f"{bytes_ / 1e9:.4f}",
                             arithmetic_intensity=f"{intensity:.2f}", error=""))

    fieldnames = ["impl", "seq_len", "time_ms", "tflops_per_s", "est_hbm_gb",
                  "arithmetic_intensity", "error"]
    meta = dict(batch=args.batch, heads=args.heads, head_dim=args.head_dim,
                dtype=args.dtype, backend=backend, device=device)
    with out_path.open("w", newline="") as f:
        f.write("# " + " ".join(f"{k}={v}" for k, v in meta.items()) + "\n")
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)
    print(f"wrote {out_path}")


if __name__ == "__main__":
    main()
