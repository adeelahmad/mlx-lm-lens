"""Base protocol and interface for ablation methods."""

from typing import Any, Dict, Protocol

import mlx.core as mx


class AblationMethod(Protocol):
  """Protocol for ablation methods.

  Each ablation method takes a layer output and transforms it in some way,
  preserving the shape. The metadata dict may contain layer-specific
  information like layer_input, attention heads, etc.
  """

  @property
  def name(self) -> str:
    """Name of the ablation method.

    Returns:
        String name (e.g., 'zero', 'mean', 'noise', 'knockout').
    """
    ...

  def apply(
    self,
    layer_output: mx.array,
    layer_index: int,
    metadata: Dict[str, Any],
  ) -> mx.array:
    """Apply ablation to layer output.

    Args:
        layer_output: Layer's output activation (shape preserved).
        layer_index: Index of the layer being ablated.
        metadata: Dict with optional layer_input, attn_weights, etc.

    Returns:
        Ablated layer output with same shape as input.
    """
    ...
