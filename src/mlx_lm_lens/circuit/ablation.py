"""Ablation sweep: measure layer importance via KL divergence."""

import logging
from typing import Any, Dict, List, Tuple

import mlx.core as mx
import numpy as np
import scipy.special

from mlx_lm_lens.ablations.registry import get_ablation
from mlx_lm_lens.utils.model_utils import (
  _get_base_model,
  _layer_forward,
  _get_embed_func,
)

logger = logging.getLogger(__name__)


class AblationSweep:
  """Sweep through layers, measure importance via KL divergence."""

  @staticmethod
  def _compute_kl_divergence(
    baseline_logits: mx.array,
    ablated_logits: mx.array,
    temperature: float = 1.0,
  ) -> float:
    """Compute KL divergence between two logit distributions.

    Args:
        baseline_logits: Original logits (vocab_size,).
        ablated_logits: Ablated logits (vocab_size,).
        temperature: Temperature for softmax. Defaults to 1.0.

    Returns:
        KL divergence (scalar).
    """
    # Numerically stable softmax with temperature
    baseline = np.array(baseline_logits)
    ablated = np.array(ablated_logits)

    # Subtract max for stability
    baseline = baseline - np.max(baseline)
    ablated = ablated - np.max(ablated)

    p = scipy.special.softmax(baseline / temperature)
    q = scipy.special.softmax(ablated / temperature)

    # KL(p||q) = sum(p * log(p/q))
    p = np.clip(p, 1e-10, 1.0)
    q = np.clip(q, 1e-10, 1.0)

    kl = np.sum(p * (np.log(p) - np.log(q)))
    return float(np.clip(kl, 0.0, 1e6))  # Clip extreme values

  @staticmethod
  def sweep_all_layers(
    model: Any,
    tokenizer: Any,
    prompt: str,
    ablation_method: str = "zero",
  ) -> Dict[str, float]:
    """Ablate each layer one at a time, measure KL divergence.

    Args:
        model: MLX model to analyze.
        tokenizer: Tokenizer for encoding (with encode method).
        prompt: Input prompt.
        ablation_method: Ablation method name ('zero', 'mean', 'noise').

    Returns:
        Dict mapping layer_key (e.g., 'layer_0') to KL divergence.
    """
    try:
      # Encode prompt
      tokens = tokenizer.encode(prompt)
      input_ids = mx.array([tokens])
      mx.eval(input_ids)

      base = _get_base_model(model)
      embed_fn = _get_embed_func(base)
      if embed_fn is None:
        raise ValueError("Cannot find embedding function")

      # Baseline forward pass
      h = embed_fn(input_ids)
      mx.eval(h)
      baseline_acts = [h]

      layers = getattr(base, "layers", [])
      for layer in layers:
        h = _layer_forward(layer, h)
        mx.eval(h)
        baseline_acts.append(h)

      # Final norm and logits
      norm = getattr(base, "norm", None)
      if norm is not None:
        h = norm(h)
        mx.eval(h)

      lm_head = getattr(model, "lm_head", None)
      if lm_head is None:
        raise ValueError("Model missing lm_head")

      baseline_logits = lm_head(h[0, -1, :])  # Last token logits
      mx.eval(baseline_logits)
      baseline_logits = np.array(baseline_logits)

      kl_divs = {}

      # Ablate each layer
      ablation = get_ablation(ablation_method)
      for layer_idx in range(len(layers)):
        try:
          # Reconstruct forward pass with ablation at layer_idx
          h = embed_fn(input_ids)
          mx.eval(h)

          for i, layer in enumerate(layers):
            if i == layer_idx:
              # Apply ablation
              h_ablated = ablation.apply(
                h,
                layer_idx,
                {"layer_input": baseline_acts[i]},
              )
              h = h_ablated
            else:
              h = _layer_forward(layer, h)
            mx.eval(h)

          # Final norm and logits
          if norm is not None:
            h = norm(h)
            mx.eval(h)

          ablated_logits = lm_head(h[0, -1, :])
          mx.eval(ablated_logits)
          ablated_logits = np.array(ablated_logits)

          kl = AblationSweep._compute_kl_divergence(
            baseline_logits, ablated_logits
          )
          kl_divs[f"layer_{layer_idx}"] = kl

        except Exception as e:
          logger.warning(f"Ablation at layer {layer_idx} failed: {e}")
          kl_divs[f"layer_{layer_idx}"] = 0.0

      return kl_divs

    except Exception as e:
      logger.error(f"Ablation sweep failed: {e}")
      return {}

  @staticmethod
  def rank_layers(
    importance: Dict[str, float],
  ) -> List[Tuple[str, float, int]]:
    """Rank layers by importance (KL divergence).

    Args:
        importance: Dict of layer_key → KL divergence.

    Returns:
        List of (layer_key, kl_value, rank) tuples, sorted by importance.
    """
    sorted_items = sorted(
      importance.items(), key=lambda x: x[1], reverse=True
    )
    return [(key, val, rank) for rank, (key, val) in enumerate(sorted_items)]
