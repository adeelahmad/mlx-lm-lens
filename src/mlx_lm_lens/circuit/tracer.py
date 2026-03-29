"""Circuit tracing: capture head importance and attention patterns."""

import logging
from dataclasses import dataclass
from typing import Any, Dict, Optional

import mlx.core as mx
import numpy as np

from mlx_lm_lens.core.activation_capture import ActivationCapture
from mlx_lm_lens.utils.model_utils import _get_attn_module, _get_base_model

logger = logging.getLogger(__name__)


@dataclass
class HeadImportanceTrace:
  """Per-layer attention head importance metrics.

  Attributes:
      head_importances: Normalized importance scores (sum to 1.0).
      raw_head_norms: Raw V-projection norms per head.
      attn_entropy: Shannon entropy of attention distribution per head.
  """

  head_importances: np.ndarray  # (n_heads,)
  raw_head_norms: np.ndarray  # (n_heads,)
  attn_entropy: np.ndarray  # (n_heads,)


class CircuitTracer:
  """Trace circuits: extract per-layer head importance."""

  @staticmethod
  def compute_head_importance(
    attn_module: Any,
    h_normed: mx.array,
    n_heads: int,
    head_dim: int,
  ) -> HeadImportanceTrace:
    """Extract per-head importance from attention module.

    Importance is based on:
    1. V-projection norms (larger norms = more important for output)
    2. Attention entropy (lower entropy = more focused attention)

    Handles GQA by broadcasting KV heads to Q heads.

    Args:
        attn_module: Attention module with v_proj, q_proj, k_proj.
        h_normed: Normalized hidden state (seq_len, hidden_dim).
        n_heads: Number of query heads.
        head_dim: Dimension per head.

    Returns:
        HeadImportanceTrace with normalized importances.
    """
    try:
      # Get V projection weights
      v_proj = getattr(attn_module, "v_proj", None)
      if v_proj is None:
        raise ValueError("Attention module missing v_proj")

      # Extract per-head norms from v_proj weight
      v_weight = v_proj.weight  # (hidden_dim, hidden_dim)
      v_weight = mx.reshape(v_weight, (n_heads, head_dim, -1))
      head_norms = mx.linalg.norm(v_weight, axis=(1, 2))
      head_norms = np.array(head_norms)

      # Compute attention entropy for each head
      q_proj = getattr(attn_module, "q_proj", None)
      if q_proj is None:
        raise ValueError("Attention module missing q_proj")

      q = q_proj(h_normed)  # (seq_len, hidden_dim)
      k_proj = getattr(attn_module, "k_proj", None)
      if k_proj is None:
        raise ValueError("Attention module missing k_proj")

      k = k_proj(h_normed)  # (seq_len, hidden_dim)

      # Reshape to (seq_len, n_heads, head_dim)
      q = mx.reshape(q, (q.shape[0], n_heads, head_dim))
      k = mx.reshape(k, (k.shape[0], -1, head_dim))  # GQA: may have fewer heads
      k_heads = k.shape[1]

      # Compute attention scores per head
      attn_entropy = np.zeros(n_heads)
      for h_idx in range(n_heads):
        q_h = q[:, h_idx, :]  # (seq_len, head_dim)
        k_h = k[:, h_idx % k_heads, :]  # Broadcast KV heads to Q heads
        scores = mx.matmul(q_h, mx.transpose(k_h))  # (seq_len, seq_len)
        attn_probs = mx.softmax(scores, axis=-1)
        attn_probs = np.array(attn_probs)

        # Shannon entropy of average attention distribution
        avg_attn = np.mean(attn_probs, axis=0)
        entropy = -np.sum(
          avg_attn * np.log(np.clip(avg_attn, 1e-10, 1.0))
        )
        attn_entropy[h_idx] = entropy

      # Combine norms and entropy for importance
      importances = head_norms * (1.0 / (attn_entropy + 1e-10))
      importances = importances / np.sum(importances)

      return HeadImportanceTrace(
        head_importances=importances,
        raw_head_norms=head_norms,
        attn_entropy=attn_entropy,
      )

    except Exception as e:
      logger.warning(f"Failed to compute head importance: {e}")
      # Return uniform importance on error
      uniform = np.ones(n_heads) / n_heads
      return HeadImportanceTrace(
        head_importances=uniform,
        raw_head_norms=np.ones(n_heads),
        attn_entropy=np.ones(n_heads),
      )

  @staticmethod
  def run_circuit_forward(
    model: Any,
    input_ids: mx.array,
  ) -> Dict[int, HeadImportanceTrace]:
    """Run forward pass capturing head importance at each layer.

    Args:
        model: MLX model to trace.
        input_ids: Input token IDs.

    Returns:
        Dict mapping layer_idx to HeadImportanceTrace.
    """
    traces = {}

    try:
      base = _get_base_model(model)
      config = getattr(base, "config", {})
      n_heads = config.get("num_attention_heads", 8)
      hidden_dim = config.get("hidden_size", 512)
      head_dim = hidden_dim // n_heads

      # Run full forward pass capturing activations
      activations = ActivationCapture.forward_with_capture(model, input_ids)

      layers = getattr(base, "layers", [])

      for i in range(len(layers)):
        try:
          layer = layers[i]
          attn = _get_attn_module(layer)

          # Get pre-attention hidden state
          h_key = f"layer_{i-1}" if i > 0 else "embed"
          h = activations.get(h_key)

          if h is not None and attn is not None:
            # Apply layer norm before attention
            norm1 = getattr(layer, "input_layernorm", None)
            if norm1 is None:
              norm1 = getattr(layer, "ln_1", None)

            h_normed = norm1(h) if norm1 is not None else h
            mx.eval(h_normed)

            trace = CircuitTracer.compute_head_importance(
              attn, h_normed, n_heads, head_dim
            )
            traces[i] = trace
        except Exception as e:
          logger.warning(f"Failed to trace layer {i}: {e}")

      return traces

    except Exception as e:
      logger.error(f"Circuit forward pass failed: {e}")
      return {}
