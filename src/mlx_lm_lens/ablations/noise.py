"""Noise ablation method."""

from typing import Any, Dict

import mlx.core as mx


class NoiseAblation:
  """Add Gaussian noise to layer output.

  Adds noise with standard deviation equal to the layer output's std,
  scaled by noise_scale factor. This measures robustness to perturbations.
  """

  def __init__(self, noise_scale: float = 1.0) -> None:
    """Initialize noise ablation.

    Args:
        noise_scale: Multiplier for noise standard deviation.
            Defaults to 1.0 (noise_std = output_std).
    """
    self.noise_scale = noise_scale

  @property
  def name(self) -> str:
    """Return ablation method name."""
    return "noise"

  def apply(
    self,
    layer_output: mx.array,
    layer_index: int,
    metadata: Dict[str, Any],
  ) -> mx.array:
    """Add Gaussian noise to layer output.

    Args:
        layer_output: Layer output to perturb.
        layer_index: Index of layer (unused).
        metadata: Optional metadata (unused).

    Returns:
        Layer output with added Gaussian noise.
    """
    # Compute standard deviation of activations
    output_std = mx.std(layer_output)

    # Generate noise with same shape
    noise = mx.random.normal(layer_output.shape)
    noise = noise * output_std * self.noise_scale

    return layer_output + noise
