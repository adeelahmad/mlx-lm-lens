"""Mean ablation method."""

from typing import Any, Dict

import mlx.core as mx


class MeanAblation:
  """Replace with mean activation across sequence.

  Computes the mean of layer output across the sequence dimension,
  then broadcasts to the original shape. This measures deviation from
  the activation's typical value.
  """

  @property
  def name(self) -> str:
    """Return ablation method name."""
    return "mean"

  def apply(
    self,
    layer_output: mx.array,
    layer_index: int,
    metadata: Dict[str, Any],
  ) -> mx.array:
    """Replace with mean activation broadcasted to original shape.

    Args:
        layer_output: Layer output (seq_len, hidden_dim) or similar.
        layer_index: Index of layer (unused).
        metadata: Optional metadata (unused).

    Returns:
        Mean activation broadcasted to match input shape.
    """
    # Compute mean across sequence dimension (axis 0)
    mean_act = mx.mean(layer_output, axis=0, keepdims=True)

    # Broadcast to full shape
    return mx.broadcast_to(mean_act, layer_output.shape)
