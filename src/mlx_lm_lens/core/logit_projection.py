"""Logit projection and layer-wise output analysis."""

import gc
import logging
from typing import Any, Dict, List, Optional, Tuple

import mlx.core as mx
import numpy as np

from mlx_lm_lens.core.activation_capture import ActivationCapture

logger = logging.getLogger(__name__)


class LogitProjector:
  """Project layer hidden states to vocabulary logits and compute top-K tokens."""

  @staticmethod
  def project_to_logits(
    hidden_state: mx.array,
    lm_head: Any,
    final_norm: Optional[Any],
    tokenizer: Any,
  ) -> Dict[int, List[Tuple[str, float, int]]]:
    """Project hidden state to vocabulary logits and extract top-K tokens.

    Args:
      hidden_state: Hidden state from a layer (seq_len, hidden_dim).
      lm_head: Language model head (unembedding layer).
      final_norm: Optional final normalization layer.
      tokenizer: Tokenizer for token-to-string conversion.

    Returns:
      Dict mapping position -> list of (token_str, prob, token_id) tuples
      sorted by probability (descending). Returns top-5 per position.

    Raises:
      ValueError: If hidden_state has invalid shape.
    """
    if hidden_state.ndim != 2:
      raise ValueError(f"Expected 2D hidden_state, got {hidden_state.ndim}D")

    seq_len = hidden_state.shape[0]
    result: Dict[int, List[Tuple[str, float, int]]] = {}

    for pos in range(seq_len):
      h = hidden_state[pos : pos + 1]

      if final_norm is not None:
        h = final_norm(h)

      logits = lm_head(h)
      mx.eval(logits)

      logits_np = np.array(logits[0])
      logits_stable = logits_np - np.max(logits_np)
      probs = np.exp(logits_stable) / np.sum(np.exp(logits_stable))

      top_indices = np.argsort(-probs)[:5]
      tokens = []

      for idx in top_indices:
        token_id = int(idx)
        prob = float(probs[idx])
        try:
          token_str = tokenizer.decode([token_id]).strip()
        except (AttributeError, ValueError, IndexError, TypeError):
          token_str = f"<unk_{token_id}>"
        tokens.append((token_str, prob, token_id))

      result[pos] = tokens

    return result

  @staticmethod
  def detect_flip_points(
    top_k_per_layer: Dict[int, Dict[int, List[Tuple[str, float, int]]]],
    position: int,
  ) -> List[Tuple[int, int, str, str]]:
    """Detect layers where top-1 token changes (flip points).

    Args:
      top_k_per_layer: Dict[layer_idx -> position -> top_k_tokens].
      position: Token position to analyze.

    Returns:
      List of (layer_from, layer_to, old_token, new_token) tuples
      for each layer where top-1 changes.
    """
    flips = []
    layers = sorted(top_k_per_layer.keys())

    for i in range(len(layers) - 1):
      curr_layer = layers[i]
      next_layer = layers[i + 1]

      if position not in top_k_per_layer[curr_layer]:
        continue
      if position not in top_k_per_layer[next_layer]:
        continue

      curr_top1 = top_k_per_layer[curr_layer][position][0][0]
      next_top1 = top_k_per_layer[next_layer][position][0][0]

      if curr_top1 != next_top1:
        flips.append((curr_layer, next_layer, curr_top1, next_top1))

    return flips

  @staticmethod
  def run_logit_lens(
    model: Any,
    tokenizer: Any,
    prompt: str,
    config: Any,
  ) -> Dict[str, Any]:
    """Run complete logit lens pipeline.

    Args:
      model: Loaded MLX model.
      tokenizer: Tokenizer for encoding/decoding.
      prompt: Input prompt text.
      config: LogitLensConfig with analysis parameters.

    Returns:
      Dict with keys:
        - "prompt": Original prompt
        - "tokens": List of (token_str, token_id) for prompt
        - "top_k_per_layer": Dict[layer -> Dict[position -> top_k]]
        - "flips": List of detected flip points
        - "config": Analysis config (top_k, position, etc.)
    """
    try:
      input_ids = tokenizer.encode(prompt)
      if isinstance(input_ids, list):
        input_ids = mx.array(input_ids).reshape(1, -1)
      else:
        input_ids = input_ids.reshape(1, -1)

      activations = ActivationCapture.forward_with_capture(model, input_ids)
      mx.eval(activations["logits"])

      seq_len = input_ids.shape[1]
      positions = list(range(seq_len)) if config.all_positions else [-1]

      positions_normalized = [
        p if p >= 0 else seq_len + p for p in positions
      ]

      top_k_per_layer: Dict[
        int, Dict[int, List[Tuple[str, float, int]]]
      ] = {}
      lm_head = getattr(model, "lm_head", None)
      final_norm = getattr(model, "norm", None)

      layer_idx = 0
      for key in sorted(activations.keys()):
        if not key.startswith("layer_"):
          continue

        hidden = activations[key]
        if hidden.ndim == 3:
          hidden = hidden[0]

        projections = LogitProjector.project_to_logits(
          hidden, lm_head, final_norm, tokenizer
        )

        top_k_per_layer[layer_idx] = projections
        layer_idx += 1

      flips_per_position = {}
      for pos in positions_normalized:
        flips = LogitProjector.detect_flip_points(top_k_per_layer, pos)
        if flips:
          flips_per_position[pos] = flips

      token_strs = [
        (tokenizer.decode([int(tid)]).strip(), int(tid))
        for tid in input_ids[0]
      ]

      return {
        "prompt": prompt,
        "tokens": token_strs,
        "top_k_per_layer": top_k_per_layer,
        "flips": flips_per_position,
        "config": {
          "top_k": config.top_k,
          "all_positions": config.all_positions,
          "positions_analyzed": positions_normalized,
        },
      }

    except Exception as e:
      logger.error(f"Logit lens pipeline failed: {e}")
      raise
