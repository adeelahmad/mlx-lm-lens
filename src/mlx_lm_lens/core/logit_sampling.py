"""Sampling and projection helpers for logit-lens generation."""

from dataclasses import dataclass
from typing import Any

import mlx.core as mx
import numpy as np


@dataclass
class TokenResult:
  """Result of a single token generation step."""

  token_num: int
  token_id: int
  token_str: str
  layer_predictions: dict[str, list[dict]]
  elapsed_sec: float
  drift_angles: dict[int, float] | None = None
  generated_text: str = ""  # Full decoded text with proper spacing


def project_to_logits(
  h: "mx.array",
  lm_head: Any | None = None,
  embed_tokens: Any | None = None,
) -> "mx.array":
  """Project hidden state to logits using lm_head or tied embeddings.

  Args:
    h: Hidden state to project.
    lm_head: Language model head (unembedding layer).
    embed_tokens: Embedding layer with as_linear method.

  Returns:
    Logits array.
  """
  if lm_head is not None:
    return lm_head(h)
  elif embed_tokens is not None and hasattr(embed_tokens, "as_linear"):
    return embed_tokens.as_linear(h)
  else:
    raise ValueError("No projection method available")


def sample_next_token(
  logits: mx.array,
  temperature: float = 1.0,
  top_p: float | None = None,
  top_k: int = 50,
  sampling_method: str = "greedy",
  rng_state: np.random.RandomState | None = None,
) -> int:
  """Sample next token from logits using specified method.

  Args:
    logits: (vocab_size,) logits array.
    temperature: Sampling temperature.
    top_p: Nucleus sampling cutoff.
    top_k: Top-K for top_k sampling.
    sampling_method: 'greedy', 'top_k', or 'nucleus'.
    rng_state: Random state for reproducibility.

  Returns:
    Next token ID.
  """
  logits_np = np.array(logits)

  if sampling_method == "greedy":
    return int(np.argmax(logits_np))

  if temperature != 1.0:
    logits_np = logits_np / temperature

  logits_stable = logits_np - np.max(logits_np)
  probs = np.exp(logits_stable) / (np.sum(np.exp(logits_stable)) + 1e-12)

  if sampling_method == "top_k":
    top_k_idx = np.argsort(-probs)[:top_k]
    probs_masked = np.zeros_like(probs)
    probs_masked[top_k_idx] = probs[top_k_idx]
    probs_masked /= (np.sum(probs_masked) + 1e-12)

    if rng_state is not None:
      return int(rng_state.choice(len(probs), p=probs_masked))
    return int(np.random.choice(len(probs), p=probs_masked))

  elif sampling_method == "nucleus":
    sorted_idx = np.argsort(-probs)
    sorted_probs = probs[sorted_idx]
    cumsum = np.cumsum(sorted_probs)

    cutoff_idx = np.where(cumsum > top_p)[0]
    if len(cutoff_idx) > 0:
      cutoff_idx = cutoff_idx[0]
      probs_masked = np.zeros_like(probs)
      probs_masked[sorted_idx[: cutoff_idx + 1]] = probs[sorted_idx[: cutoff_idx + 1]]
    else:
      probs_masked = probs.copy()

    probs_masked /= (np.sum(probs_masked) + 1e-12)

    if rng_state is not None:
      return int(rng_state.choice(len(probs), p=probs_masked))
    return int(np.random.choice(len(probs), p=probs_masked))

  return int(np.argmax(logits_np))
