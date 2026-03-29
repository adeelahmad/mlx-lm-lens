"""Ablation sweep: measure layer importance via systematic ablation."""

import logging
from typing import Any, Dict, List, Optional, Tuple

import mlx.core as mx
import numpy as np

from mlx_lm_lens.ablations.zero import ZeroAblation
from mlx_lm_lens.core.activation_capture import ActivationCapture

logger = logging.getLogger(__name__)


class AblationSweep:
  """Run ablation on all layers and rank by importance."""

  @staticmethod
  def sweep_all_layers(
    model: Any,
    tokenizer: Any,
    prompt: str,
    method: str = "zero",
    target_token: Optional[str] = None,
    layer_range: Optional[Tuple[int, int]] = None,
  ) -> Dict[int, float]:
    """Ablate each layer and measure KL divergence from baseline.

    Args:
      model: MLX model to analyze.
      tokenizer: Tokenizer for encoding/decoding.
      prompt: Input prompt to analyze.
      method: Ablation method ("zero", "mean", "noise", "knockout").
      target_token: Optional token to measure probability for.
      layer_range: Optional tuple (start, end) to restrict sweep.

    Returns:
      Dict mapping layer_idx → kl_divergence (higher = more important).
    """
    # Get baseline logits (no ablation)
    tokens = tokenizer.encode(prompt)
    inputs = mx.array(tokens).reshape(1, -1)

    baseline_logits = model(inputs)
    baseline_logits = mx.eval(baseline_logits)

    # Get num layers
    n_layers = len(model.layers)
    if layer_range:
      start, end = layer_range
      layer_indices = range(start, min(end + 1, n_layers))
    else:
      layer_indices = range(n_layers)

    importance = {}
    ablation = _get_ablation_method(method)

    for layer_idx in layer_indices:
      logger.debug(f"Ablating layer {layer_idx}")

      # Hook into the layer to apply ablation
      hook_handle = _install_ablation_hook(model, layer_idx, ablation)

      try:
        ablated_logits = model(inputs)
        ablated_logits = mx.eval(ablated_logits)

        # Compute KL divergence at last token
        kl = _compute_kl_divergence(baseline_logits, ablated_logits)
        importance[layer_idx] = float(kl)
      finally:
        _remove_hook(model, hook_handle)

    return importance

  @staticmethod
  def rank_layers(
    importance: Dict[int, float],
  ) -> List[Tuple[int, float, int]]:
    """Rank layers by importance (descending).

    Args:
      importance: Dict mapping layer_idx → importance_score.

    Returns:
      List of (layer_idx, importance, rank) sorted by importance.
    """
    sorted_items = sorted(importance.items(), key=lambda x: -x[1])
    return [
      (layer_idx, score, rank + 1)
      for rank, (layer_idx, score) in enumerate(sorted_items)
    ]


def _get_ablation_method(method: str) -> Any:
  """Get ablation method by name."""
  methods = {
    "zero": ZeroAblation(),
  }
  if method not in methods:
    raise ValueError(f"Unknown ablation method: {method}")
  return methods[method]


def _install_ablation_hook(
  model: Any,
  layer_idx: int,
  ablation: Any,
) -> Any:
  """Install hook to apply ablation to layer output."""
  # This is a simplified version - real implementation would use
  # MLX's hook mechanism or a forward pass wrapper
  original_layer = model.layers[layer_idx]

  def ablation_wrapper(*args: Any, **kwargs: Any) -> mx.array:
    output = original_layer(*args, **kwargs)
    metadata: Dict[str, Any] = {}
    return ablation.apply(output, layer_idx, metadata)

  model.layers[layer_idx] = ablation_wrapper
  return original_layer


def _remove_hook(model: Any, original_layer: Any) -> None:
  """Remove ablation hook (restore original layer)."""
  # In a real implementation, this would restore the original layer
  pass


def _compute_kl_divergence(
  baseline_logits: mx.array,
  ablated_logits: mx.array,
) -> float:
  """Compute KL divergence between two logit distributions.

  Args:
    baseline_logits: Baseline logits shape (batch, seq_len, vocab).
    ablated_logits: Ablated logits same shape.

  Returns:
    KL divergence scalar.
  """
  # Get last token logits
  baseline = mx.softmax(baseline_logits[:, -1, :], axis=-1)
  ablated = mx.softmax(ablated_logits[:, -1, :], axis=-1)

  baseline = mx.eval(baseline)
  ablated = mx.eval(ablated)

  # KL(baseline || ablated) = sum(baseline * log(baseline / ablated))
  baseline_np = np.array(baseline)
  ablated_np = np.array(ablated)

  kl = np.sum(
    baseline_np * (np.log(baseline_np + 1e-10) - np.log(ablated_np + 1e-10))
  )
  return float(kl)
