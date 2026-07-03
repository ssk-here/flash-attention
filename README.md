# FlashAttention from Scratch in JAX Pallas

**A three-way performance study of attention: eager JAX vs. XLA-compiled vs. a hand-written fused Pallas kernel.**

This project implements the FlashAttention forward pass as a custom accelerator kernel using
[JAX Pallas](https://docs.jax.dev/en/latest/pallas/index.html), and benchmarks it against two
baselines to make a quantitative engineering argument about the **memory wall** in attention.

| # | Implementation | What it demonstrates |
|---|----------------|----------------------|
| 1 | `naive_attention` — plain JAX, eager | Materializes the full `(N, N)` score matrix in HBM. Shows the O(N²) memory cliff. |
| 2 | `attention_xla` — same math under `jax.jit` | What the XLA compiler gives you for free (automatic elementwise fusion). The honest baseline. |
| 3 | `flash_attention` — hand-written Pallas kernel | Tiling + online softmax + kernel fusion. The score matrix never exists in HBM. |

The same Pallas kernel runs on **GPU** (Triton backend), **TPU** (Mosaic backend), and on
**CPU via interpret mode** (slow, but lets you validate correctness with zero accelerator cost).

## Why attention is memory-bound

Standard attention computes `softmax(QKᵀ / √d) V`. For sequence length N and head dimension d:

- **FLOPs:** `4·N²·d` per head (two matmuls) — grows with d.
- **HBM traffic:** dominated by writing and re-reading the `(N, N)` score and probability
  matrices — independent of d.

The ratio (arithmetic intensity, FLOPs/byte) is low, so on any modern accelerator the matrix
units sit idle waiting on HBM. FlashAttention does **slightly more FLOPs** (rescaling) but moves
**far fewer bytes**: Q-blocks stay resident in fast on-chip memory (VMEM on TPU, SRAM on GPU)
while K/V blocks stream past, and an online-softmax recurrence maintains running row maxima
`m` and normalizers `l` so no intermediate ever spills to HBM. This raises arithmetic intensity
past the roofline ridge point — the kernel becomes compute-bound at long sequence lengths.

The online softmax recurrence per incoming K/V block:

```
m_new = max(m_old, rowmax(S_block))
α     = exp(m_old − m_new)                 # rescale factor for history
P     = exp(S_block − m_new)
l_new = α·l_old + rowsum(P)
acc   = α·acc + P·V_block
out   = acc / l_new                        # after the last block
```

## Repository layout

```
flash_attention/
  naive.py           # eager reference implementation
  pallas_kernel.py   # fused FlashAttention forward kernel (Pallas)
  __init__.py        # exports naive_attention, attention_xla, flash_attention
tests/
  test_correctness.py  # allclose vs. reference — must be green before benchmarking
benchmarks/
  run_benchmark.py   # sweeps sequence lengths, writes CSV (time, TFLOP/s, intensity)
  plot_results.py    # runtime / throughput / roofline plots from the CSV
notebooks/
  colab_runner.ipynb # clone → install → test → benchmark → plot, on Colab
```

## Quickstart (Colab)

Open `notebooks/colab_runner.ipynb` in Google Colab, pick a runtime
(**Runtime → Change runtime type**):

- **CPU (free):** validates correctness via Pallas interpret mode. No compute units burned.
- **T4 (free):** correctness (interpret) plus a *real* naive-vs-XLA GPU
  benchmark. The Pallas kernel itself can't compile here — JAX's Triton
  backend requires Ampere (compute capability 8.0+) and the T4 is Turing
  (7.5). The library detects this and falls back to interpret mode.
- **L4 (Pro):** full three-way GPU benchmark through the Triton backend.
- **TPU:** full three-way TPU benchmark through the Mosaic backend.

Or locally / in a terminal:

```bash
pip install -e .
python -m pytest tests -q                                   # correctness first
python benchmarks/run_benchmark.py --seq-lens 512 1024 2048 4096 \
    --dtype bf16 --repeats 10 --seed 0 --output results/results.csv
python benchmarks/plot_results.py results/results.csv --device L4-bf16-tensor
```

## Benchmark methodology

- Each configuration is measured over `--repeats` independent random data draws
  (default 10, seeded: repeat *r* uses `PRNGKey(seed + r)`, so runs are exactly
  reproducible). Within a repeat all three implementations see the same tensors.
  Plots show median across repeats with ±1σ error bars.
- Per draw: warmup iterations to exclude compilation, then median of timed runs with
  `jax.block_until_ready` (JAX dispatch is async — forgetting this measures nothing).
- FLOPs counted analytically: `4·B·H·N²·d`.
- HBM bytes estimated analytically per implementation (naive pays ~4·B·H·N² elements of
  score-matrix traffic; flash pays only Q/K/V/O).
- Roofline: achieved TFLOP/s vs. arithmetic intensity, against published peaks
  (NVIDIA L4: 121 TF bf16 tensor / 300 GB/s; TPU v5e: 197 TF bf16 / 819 GB/s).

## Results

*(populated after running on L4 / TPU v5e — placeholder)*

| seq len | naive (ms) | XLA jit (ms) | Pallas (ms) | speedup vs jit |
|--------:|-----------:|-------------:|------------:|---------------:|
| 1024    |            |              |             |                |
| 2048    |            |              |             |                |
| 4096    |            |              |             |                |

## Scope & roadmap

Implemented: forward pass, non-causal, fp32/bf16, block sizes aligned to TPU tiling (128).

Future work, in order of interest:
- Causal masking with block-level skip (upper-triangular K/V blocks never loaded)
- Backward pass via recomputation (the FlashAttention gradient trick)
- Block-size autotuning per device
- Comparison against `jax.nn.dot_product_attention` and cuDNN flash attention

## References

- Dao et al., *FlashAttention: Fast and Memory-Efficient Exact Attention with IO-Awareness* (2022)
- Dao, *FlashAttention-2* (2023)
- Milakov & Gimelshein, *Online normalizer calculation for softmax* (2018)
- [JAX Pallas documentation](https://docs.jax.dev/en/latest/pallas/index.html)
