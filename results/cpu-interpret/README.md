# CPU interpret-mode validation run (NOT a benchmark)

Artifacts from the first Colab run on a **free CPU runtime** (JAX 0.7.2,
2026-07-03). Pallas executes in interpret mode here, so the timings are
meaningless as performance numbers — this run exists to prove the pipeline:

- all 7 correctness tests passed on Colab,
- the benchmark sweep, CSV output, and all three plots ran end-to-end.

Note the `arithmetic_intensity` column is already telling the real story
even on CPU: the Pallas kernel's AI grows with sequence length (64 at
N=256, 128 at N=512) while naive/XLA stays flat at 12.8 — the roofline
argument in data form.

Real numbers (L4 GPU / TPU v5e, bf16, 10 repeats) land in `results/` when
measured.
