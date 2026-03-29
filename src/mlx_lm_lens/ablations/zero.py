"""Zero ablation method."""

from typing import Any, Dict

import mlx.core as mx


class ZeroAblation:
  """Zero out the entire layer output.

  This measures the total contribution of the layer to the model's
  output by completely removing its effect on the residual stream.
  """

  @property
  def name(self) -> str:
    """Return ablation method name."""
    return "zero"

  def apply(
    self,
    layer_output: mx.array,
    layer_index: int,
    metadata: Dict[str, Any],
  ) -> mx.array:
    """Zero out layer output.

    Args:
        layer_output: Layer output to ablate.
        layer_index: Index of layer (unused).
        metadata: Optional metadata (unused).

    Returns:
        Zeros with same shape as layer_output.
    """
    return mx.zeros_like(layer_output)
