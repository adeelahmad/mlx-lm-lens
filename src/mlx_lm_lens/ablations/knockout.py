"""Residual knockout ablation method."""

from typing import Any, Dict

import mlx.core as mx


class ResidualKnockout:
  """Remove only residual addition, keep skip connection.

  This ablation preserves the skip connection (layer_input) but removes
  the layer's additive contribution. Requires layer_input in metadata.
  """

  @property
  def name(self) -> str:
    """Return ablation method name."""
    return "knockout"

  def apply(
    self,
    layer_output: mx.array,
    layer_index: int,
    metadata: Dict[str, Any],
  ) -> mx.array:
    """Keep skip connection, remove residual addition.

    In typical transformer residual blocks:
      layer_output = layer_input + mlp_or_attn_output

    This returns just layer_input (the skip connection).

    Args:
        layer_output: Full layer output (including skip connection).
        layer_index: Index of layer (unused).
        metadata: Dict containing 'layer_input' key.

    Returns:
        Just the layer_input (skip connection).

    Raises:
        KeyError: If 'layer_input' not in metadata.
    """
    if "layer_input" not in metadata:
      raise KeyError(
        f"ResidualKnockout requires 'layer_input' in metadata "
        f"for layer {layer_index}"
      )

    return metadata["layer_input"]
