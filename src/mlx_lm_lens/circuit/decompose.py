"""Residual stream decomposition: per-layer contribution analysis."""

import logging
from typing import Any, Dict

import mlx.core as mx
import numpy as np

from mlx_lm_lens.utils.model_utils import _get_base_model, _get_embed_func

logger = logging.getLogger(__name__)


class ResidualStreamDecomposer:
  """Decompose residual stream into per-layer additive contributions."""

  @staticmethod
  def decompose_residual(
    model: Any,
    tokenizer: Any,
    prompt: str,
    position: int = -1,
  ) -> Dict[int, float]:
    """Decompose residual stream at a token position.

    Runs forward pass, collects hidden states at each layer,
    and computes h_i - h_{i-1} (residual contribution per layer).

    Args:
      model: MLX model to analyze.
      tokenizer: Tokenizer for encoding prompt.
      prompt: Input prompt.
      position: Token position (-1 = last, or 0-indexed position).

    Returns:
      Dict mapping layer_idx → L2 norm of that layer's contribution.
    """
    tokens = tokenizer.encode(prompt)
    input_ids = mx.array([tokens])
    mx.eval(input_ids)

    base_model = _get_base_model(model)
    embed_func = _get_embed_func(base_model)

    if embed_func is None:
      raise ValueError("Cannot find embedding function")

    # Get initial embeddings
    h = embed_func(input_ids)
    mx.eval(h)

    # Normalize position index
    if position == -1:
      position = h.shape[1] - 1
    elif position < 0 or position >= h.shape[1]:
      raise ValueError(f"Invalid position {position} for sequence length {h.shape[1]}")

    contributions = {}
    layers = getattr(base_model, "layers", [])

    prev_h = h
    for layer_idx, layer in enumerate(layers):
      # Forward through layer
      h = _layer_forward(layer, h)
      mx.eval(h)

      # Compute residual: h_i - h_{i-1}
      residual = h - prev_h
      mx.eval(residual)

      # Get L2 norm at target position
      residual_np = np.array(residual[0, position, :])
      norm = float(np.linalg.norm(residual_np, ord=2))
      contributions[layer_idx] = norm

      prev_h = h

    return contributions

  @staticmethod
  def show_contributions(
    model: Any,
    tokenizer: Any,
    prompt: str,
    position: int = -1,
  ) -> Dict[str, Any]:
    """Get per-layer contributions with normalized scores.

    Args:
      model: MLX model.
      tokenizer: Tokenizer.
      prompt: Input prompt.
      position: Token position.

    Returns:
      Dict with raw norms and normalized scores.
    """
    contributions = ResidualStreamDecomposer.decompose_residual(
      model,
      tokenizer,
      prompt,
      position,
    )

    # Normalize to [0, 1]
    total = sum(contributions.values())
    if total == 0:
      normalized = {k: 0.0 for k in contributions}
    else:
      normalized = {k: v / total for k, v in contributions.items()}

    return {
      "raw_norms": contributions,
      "normalized_scores": normalized,
    }


def _layer_forward(layer: Any, h: mx.array) -> mx.array:
  """Generic layer forward pass."""
  if callable(layer):
    return layer(h)
  raise ValueError(f"Layer {layer} is not callable")
