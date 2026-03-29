"""Capture activations during forward pass without modifying model."""

from __future__ import annotations

import logging
from typing import TYPE_CHECKING, Any

import mlx.core as mx

if TYPE_CHECKING:
  from mlx_lm_lens.core.drift_correction import DriftCorrector

from mlx_lm_lens.utils.model_utils import (
  _get_base_model,
  _get_embed_func,
  _layer_forward,
)

logger = logging.getLogger(__name__)


class ActivationCaptureError(Exception):
  """Raised when activation capture fails."""

  pass


class ActivationCapture:
  """Capture hidden states from model forward pass."""

  @staticmethod
  def forward_with_capture(
    model: Any,
    input_ids: mx.array,
    attention_mask: mx.array | None = None,
    drift_corrector: DriftCorrector | None = None,
  ) -> dict[str, mx.array]:
    """Forward pass through model capturing activations at each layer.

    Args:
      model: MLX model to forward.
      input_ids: Input token IDs (seq_len,) or (batch, seq_len).
      attention_mask: Optional attention mask.

    Returns:
      Dictionary with keys:
        - "embed": Embedding output (seq_len, hidden_dim)
        - "layer_N": Layer N hidden state
        - "final_norm": Final normalization output
        - "logits": Final logits (seq_len, vocab_size)

    Raises:
      ActivationCaptureError: If capture fails at any step.
    """
    try:
      base = _get_base_model(model)
      embed_fn = _get_embed_func(base)

      if embed_fn is None:
        raise ActivationCaptureError("Cannot find embedding function")

      # Get embeddings
      h = embed_fn(input_ids)
      mx.eval(h)
      activations: dict[str, mx.array] = {"embed": h}

      # Forward through each layer
      layers = getattr(base, "layers", [])
      for i, layer in enumerate(layers):
        try:
          h = _layer_forward(layer, h, mask=attention_mask)
          mx.eval(h)
          if drift_corrector is not None:
            h = drift_corrector.apply(layer_idx=i, hidden=h)
          activations[f"layer_{i}"] = h
        except Exception as e:
          raise ActivationCaptureError(
            f"Layer {i} forward failed: {e}"
          ) from e

      # Final normalization
      norm = getattr(base, "norm", None)
      if norm is not None:
        try:
          h = norm(h)
          mx.eval(h)
          activations["final_norm"] = h
        except Exception as e:
          logger.warning(f"Final norm failed: {e}")
          activations["final_norm"] = h

      # Logits projection
      lm_head = getattr(base, "lm_head", None) or getattr(
        model, "lm_head", None
      )
      if lm_head is not None:
        try:
          logits = lm_head(h)
          mx.eval(logits)
          activations["logits"] = logits
        except Exception as e:
          logger.warning(f"LM head projection failed: {e}")

      return activations

    except ActivationCaptureError:
      raise
    except Exception as e:
      raise ActivationCaptureError(f"Forward pass failed: {e}") from e
