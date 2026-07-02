"""Three implementations of the same attention math.

- naive_attention: eager JAX, materializes the (N, N) score matrix in HBM.
- attention_xla:   identical math under jax.jit — XLA's automatic fusion.
- flash_attention: hand-written fused Pallas kernel (tiling + online softmax).
"""

import jax

from flash_attention.naive import naive_attention
from flash_attention.pallas_kernel import flash_attention

attention_xla = jax.jit(naive_attention)

__all__ = ["naive_attention", "attention_xla", "flash_attention"]
