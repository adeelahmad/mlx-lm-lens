"""Activation patching: swap layer outputs between models."""

import logging
from typing import Any, Dict

import mlx.core as mx
import numpy as np
import scipy.special

from mlx_lm_lens.utils.model_utils import (
  _get_base_model,
  _layer_forward,
  _get_embed_func,
)

logger = logging.getLogger(__name__)


class ActivationPatcher:
  """Patch layer activations between two models."""

  @staticmethod
  def _compute_kl_divergence(
    baseline: np.ndarray,
    patched: np.ndarray,
  ) -> float:
    """Compute KL divergence between logit distributions.

    Args:
        baseline: Baseline logits (vocab_size,).
        patched: Patched logits (vocab_size,).

    Returns:
        KL divergence value.
    """
    baseline = baseline - np.max(baseline)
    patched = patched - np.max(patched)

    p = scipy.special.softmax(baseline)
    q = scipy.special.softmax(patched)

    p = np.clip(p, 1e-10, 1.0)
    q = np.clip(q, 1e-10, 1.0)

    kl = np.sum(p * (np.log(p) - np.log(q)))
    return float(np.clip(kl, 0.0, 1e6))

  @staticmethod
  def patch_layer(
    model_a: Any,
    model_b: Any,
    tokenizer: Any,
    prompt: str,
    patch_layer_idx: int,
  ) -> Dict[str, Any]:
    """Replace layer_i output in model_A with model_B's layer_i.

    Args:
        model_a: Base model (get final output from this).
        model_b: Donor model (get layer from this).
        tokenizer: Tokenizer for prompt encoding.
        prompt: Input prompt.
        patch_layer_idx: Layer index to patch.

    Returns:
        Dict with baseline_logits, patched_logits, kl, token_flip.
    """
    try:
      tokens = tokenizer.encode(prompt)
      input_ids = mx.array([tokens])
      mx.eval(input_ids)

      base_a = _get_base_model(model_a)
      base_b = _get_base_model(model_b)

      embed_a = _get_embed_func(base_a)
      embed_b = _get_embed_func(base_b)

      if embed_a is None or embed_b is None:
        raise ValueError("Cannot find embedding functions")

      # Baseline: forward through model_a
      h_a = embed_a(input_ids)
      mx.eval(h_a)
      layers_a = getattr(base_a, "layers", [])

      for i, layer in enumerate(layers_a):
        h_a = _layer_forward(layer, h_a)
        mx.eval(h_a)

      norm_a = getattr(base_a, "norm", None)
      if norm_a is not None:
        h_a = norm_a(h_a)
        mx.eval(h_a)

      lm_head_a = getattr(model_a, "lm_head", None)
      if lm_head_a is None:
        raise ValueError("Model A missing lm_head")

      baseline_logits = lm_head_a(h_a[0, -1, :])
      mx.eval(baseline_logits)
      baseline_logits = np.array(baseline_logits)

      # Patched: forward through model_b up to patch_layer_idx,
      # then continue with model_a
      h_b = embed_b(input_ids)
      mx.eval(h_b)
      layers_b = getattr(base_b, "layers", [])

      for i in range(patch_layer_idx + 1):
        if i < len(layers_b):
          h_b = _layer_forward(layers_b[i], h_b)
          mx.eval(h_b)

      # Continue with model_a from next layer
      for i in range(patch_layer_idx + 1, len(layers_a)):
        h_b = _layer_forward(layers_a[i], h_b)
        mx.eval(h_b)

      norm_a = getattr(base_a, "norm", None)
      if norm_a is not None:
        h_b = norm_a(h_b)
        mx.eval(h_b)

      patched_logits = lm_head_a(h_b[0, -1, :])
      mx.eval(patched_logits)
      patched_logits = np.array(patched_logits)

      kl = ActivationPatcher._compute_kl_divergence(
        baseline_logits, patched_logits
      )

      # Detect token flip
      baseline_top = np.argmax(baseline_logits)
      patched_top = np.argmax(patched_logits)
      token_flip = baseline_top != patched_top

      return {
        "baseline_logits": baseline_logits,
        "patched_logits": patched_logits,
        "kl": kl,
        "token_flip": token_flip,
      }

    except Exception as e:
      logger.error(f"Patching at layer {patch_layer_idx} failed: {e}")
      return {
        "baseline_logits": np.array([]),
        "patched_logits": np.array([]),
        "kl": 0.0,
        "token_flip": False,
      }

  @staticmethod
  def sweep_patch_all_layers(
    model_a: Any,
    model_b: Any,
    tokenizer: Any,
    prompt: str,
  ) -> Dict[int, float]:
    """Patch each layer one at a time, measure KL divergence.

    Args:
        model_a: Base model.
        model_b: Donor model.
        tokenizer: Tokenizer for prompt encoding.
        prompt: Input prompt.

    Returns:
        Dict mapping layer_idx to KL divergence.
    """
    try:
      base_a = _get_base_model(model_a)
      layers_a = getattr(base_a, "layers", [])

      kl_divs = {}
      for layer_idx in range(len(layers_a)):
        result = ActivationPatcher.patch_layer(
          model_a, model_b, tokenizer, prompt, layer_idx
        )
        kl_divs[layer_idx] = result.get("kl", 0.0)

      return kl_divs

    except Exception as e:
      logger.error(f"Patch sweep failed: {e}")
      return {}
