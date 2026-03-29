"""Compute structured diffs between base and adapted logit-lens results."""

import logging
from typing import Any

logger = logging.getLogger(__name__)


def _diff_position(
    base_tokens: list[tuple[str, float, int]],
    adapted_tokens: list[tuple[str, float, int]],
) -> dict[str, Any]:
  """Compute per-position diff from two top-k token lists.

  Args:
    base_tokens: List of (token_str, prob, token_id) from base model.
    adapted_tokens: List of (token_str, prob, token_id) from adapted model.

  Returns:
    Dict with base_top1, adapted_top1, top1_changed, shared_tokens, base_only, adapted_only.
  """
  if not base_tokens or not adapted_tokens:
    return {
      "base_top1": {"token": "", "prob": 0.0},
      "adapted_top1": {"token": "", "prob": 0.0},
      "top1_changed": False,
      "shared_tokens": [],
      "base_only": [],
      "adapted_only": [],
    }

  base_top1 = base_tokens[0]
  adapted_top1 = adapted_tokens[0]

  result: dict[str, Any] = {
    "base_top1": {"token": base_top1[0], "prob": float(base_top1[1])},
    "adapted_top1": {"token": adapted_top1[0], "prob": float(adapted_top1[1])},
    "top1_changed": base_top1[0] != adapted_top1[0],
  }

  base_dict = {tok[0]: tok[1] for tok in base_tokens}
  adapted_dict = {tok[0]: tok[1] for tok in adapted_tokens}

  base_set = set(base_dict.keys())
  adapted_set = set(adapted_dict.keys())

  shared = base_set & adapted_set
  result["shared_tokens"] = [
    {
      "token": tok,
      "base_prob": float(base_dict[tok]),
      "adapted_prob": float(adapted_dict[tok]),
      "delta": float(adapted_dict[tok] - base_dict[tok]),
    }
    for tok in sorted(shared)
  ]

  result["base_only"] = [
    {"token": tok, "prob": float(base_dict[tok])} for tok in sorted(base_set - adapted_set)
  ]

  result["adapted_only"] = [
    {"token": tok, "prob": float(adapted_dict[tok])} for tok in sorted(adapted_set - base_set)
  ]

  return result


def compute_diff(
    base_result: dict[str, Any],
    adapted_result: dict[str, Any],
    model_path: str,
    adapter_path: str,
) -> dict[str, Any]:
  """Compute structured diff between base and adapted logit-lens results.

  Args:
    base_result: Output of LogitProjector.run_logit_lens() on base model.
    adapted_result: Output of LogitProjector.run_logit_lens() on adapted model.
    model_path: Model path, recorded in diff metadata.
    adapter_path: Adapter path, recorded in diff metadata.

  Returns:
    Diff dict with prompt, config, and layers containing per-position diffs.
  """
  base_tokens = base_result.get("tokens", [])
  adapted_tokens = adapted_result.get("tokens", [])

  if base_tokens != adapted_tokens:
    logger.warning("Prompt tokenization differs between base and adapted models")

  base_top_k = base_result.get("top_k_per_layer", {})
  adapted_top_k = adapted_result.get("top_k_per_layer", {})

  layers_diff: dict[str, dict[str, Any]] = {}

  all_layers = set(base_top_k.keys()) | set(adapted_top_k.keys())

  if len(all_layers) != len(base_top_k) or len(all_layers) != len(adapted_top_k):
    logger.warning(
      f"Layer count mismatch: base {len(base_top_k)} layers, "
      f"adapted {len(adapted_top_k)} layers, union {len(all_layers)}"
    )

  for layer_idx in sorted(all_layers):
    layer_key = str(layer_idx)
    base_layer = base_top_k.get(layer_idx, {})
    adapted_layer = adapted_top_k.get(layer_idx, {})

    positions_diff: dict[str, Any] = {}

    all_positions = set(base_layer.keys()) | set(adapted_layer.keys())

    for pos in sorted(all_positions):
      base_tokens_at_pos = base_layer.get(pos, [])
      adapted_tokens_at_pos = adapted_layer.get(pos, [])
      positions_diff[str(pos)] = _diff_position(base_tokens_at_pos, adapted_tokens_at_pos)

    layers_diff[layer_key] = positions_diff

  return {
    "prompt": base_result.get("prompt", ""),
    "config": {
      "top_k": base_result.get("config", {}).get("top_k", 5),
      "model_path": model_path,
      "adapter_path": adapter_path,
    },
    "layers": layers_diff,
  }
