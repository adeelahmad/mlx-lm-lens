"""Generate tokens and log per-layer predictions."""

from __future__ import annotations

import gc
import logging
import time
from collections.abc import Generator
from typing import TYPE_CHECKING, Any

import mlx.core as mx
import numpy as np

from mlx_lm_lens.core.activation_capture import ActivationCapture
from mlx_lm_lens.core.logit_sampling import TokenResult, project_to_logits, sample_next_token
from mlx_lm_lens.exceptions import ActivationCaptureError
from mlx_lm_lens.utils.model_utils import _get_base_model

if TYPE_CHECKING:
  from mlx_lm_lens.core.drift_correction import DriftCorrector

logger = logging.getLogger(__name__)


class LogitLensGenerator:
  """Generate tokens and capture per-layer predictions (deprecated wrapper)."""

  @staticmethod
  def generate_with_logit_lens(
    model: Any,
    tokenizer: Any,
    prompt: str,
    max_tokens: int = 100,
    log_from_token: int = 0,
    top_k: int = 5,
    temperature: float = 1.0,
    top_p: float | None = None,
    sampling_method: str = "greedy",
    seed: int | None = None,
    include_prompt: bool = False,
    show_progress: bool = True,
    stop_strings: list[str] | None = None,
  ) -> dict[str, Any]:
    """Backward-compatible wrapper for generate_with_logit_lens function."""
    return generate_with_logit_lens(
      model,
      tokenizer,
      prompt,
      max_tokens=max_tokens,
      log_from_token=log_from_token,
      top_k=top_k,
      temperature=temperature,
      top_p=top_p,
      sampling_method=sampling_method,
      seed=seed,
      stop_strings=stop_strings,
    )


def stream_with_logit_lens(
  model: Any,
  tokenizer: Any,
  prompt: str,
  max_tokens: int = 100,
  log_from_token: int = 0,
  top_k: int = 5,
  temperature: float = 1.0,
  top_p: float | None = None,
  sampling_method: str = "greedy",
  seed: int | None = None,
  stop_strings: list[str] | None = None,
  drift_corrector: DriftCorrector | None = None,
) -> Generator[TokenResult, None, None]:
  """Stream tokens with per-layer predictions.

  Yields one TokenResult per generated token. Respects stop_strings:
  if any stop_string is found in the decoded token, stops cleanly.

  Args:
    model: Loaded MLX model.
    tokenizer: Tokenizer for encoding/decoding.
    prompt: Initial prompt text.
    max_tokens: Maximum tokens to generate.
    log_from_token: Start logging from this token index.
    top_k: Number of top predictions per layer.
    temperature: Sampling temperature (>0).
    top_p: Nucleus sampling cutoff (0-1).
    sampling_method: 'greedy', 'top_k', or 'nucleus'.
    seed: Random seed for reproducibility.
    stop_strings: Strings that trigger early termination.

  Yields:
    TokenResult objects (one per token generated).
  """
  try:
    rng_state = None
    if seed is not None:
      rng_state = np.random.RandomState(seed)
      logger.info(f"Using seed: {seed}")

    prompt_ids = tokenizer.encode(prompt)
    if not isinstance(prompt_ids, list):
      prompt_ids = prompt_ids.tolist()

    current_ids = prompt_ids.copy()
    prompt_len = len(prompt_ids)
    start_time = time.monotonic()

    logger.info(f"Starting generation with prompt: '{prompt}'")
    logger.info(
      f"Config: max_tokens={max_tokens}, log_from={log_from_token}, "
      f"temp={temperature}, top_p={top_p}, method={sampling_method}"
    )
    logger.info(f"Prompt tokens: {prompt_len}")

    for token_num in range(max_tokens):
      if drift_corrector is not None:
        drift_corrector.advance_token()

      input_array = mx.array([current_ids])
      activations = ActivationCapture.forward_with_capture(
        model, input_array, drift_corrector=drift_corrector
      )

      base_model = _get_base_model(model)
      final_norm = getattr(base_model, "norm", None)
      lm_head = getattr(base_model, "lm_head", None) or getattr(model, "lm_head", None)
      embed_tokens = getattr(base_model, "embed_tokens", None)

      if lm_head is None and (embed_tokens is None or not hasattr(embed_tokens, "as_linear")):
        raise ActivationCaptureError("No logits projection found")

      layer_predictions = {}
      if token_num >= log_from_token:
        for layer_key in sorted(activations.keys()):
          if not layer_key.startswith("layer_"):
            continue

          hidden = activations[layer_key]
          if hidden.ndim == 3:
            hidden = hidden[0]

          h = hidden[-1:] if hidden.ndim > 1 else hidden.reshape(1, -1)
          if final_norm is not None and callable(final_norm):
            h = final_norm(h)

          logits = project_to_logits(h, lm_head=lm_head, embed_tokens=embed_tokens)
          mx.eval(logits)

          if logits.dtype == mx.bfloat16:
            logits = logits.astype(mx.float32)
            mx.eval(logits)
          if logits.ndim == 3:
            logits_np = np.array(logits[0, -1])
          else:
            logits_np = np.array(logits[-1])

          logits_stable = logits_np - np.max(logits_np)
          probs = np.exp(logits_stable) / (np.sum(np.exp(logits_stable)) + 1e-12)

          top_indices = np.argsort(-probs)[:top_k]
          predictions = []
          for idx in top_indices:
            token_id = int(idx)
            prob = float(probs[idx])
            try:
              token_str = tokenizer.decode([token_id]).strip()
            except Exception:
              token_str = f"<unk_{token_id}>"
            predictions.append({"token": token_str, "token_id": token_id, "probability": prob})

          layer_predictions[layer_key] = predictions

      # Get next token
      last_layer_key = max(
        [k for k in activations.keys() if k.startswith("layer_")],
        key=lambda x: int(x.split("_")[1]),
      )

      hidden = activations[last_layer_key]
      if hidden.ndim == 3:
        hidden = hidden[0]

      h = hidden[-1:] if hidden.ndim > 1 else hidden.reshape(1, -1)
      if final_norm is not None:
        h = final_norm(h)

      logits = project_to_logits(h, lm_head=lm_head, embed_tokens=embed_tokens)
      mx.eval(logits)

      if logits.dtype == mx.bfloat16:
        logits = logits.astype(mx.float32)
        mx.eval(logits)
      if logits.ndim == 3:
        logits_flat = np.array(logits[0, -1, :])
      else:
        logits_flat = np.array(logits[-1, :])

      next_token_id = sample_next_token(
        logits_flat,
        temperature=temperature,
        top_p=top_p,
        top_k=5,
        sampling_method=sampling_method,
        rng_state=rng_state,
      )

      next_token_str = tokenizer.decode([next_token_id]).strip()
      current_ids.append(next_token_id)

      # Decode full sequence to preserve spacing
      full_generated_text = tokenizer.decode(current_ids[prompt_len:])

      elapsed = time.monotonic() - start_time
      drift_angles = None
      if drift_corrector is not None and drift_corrector.phase == "correcting":
        drift_angles = drift_corrector.get_last_angles()

      yield TokenResult(
        token_num=token_num,
        token_id=next_token_id,
        token_str=next_token_str,
        layer_predictions=layer_predictions,
        elapsed_sec=elapsed,
        drift_angles=drift_angles,
        generated_text=full_generated_text,
      )

      if stop_strings and any(s in next_token_str for s in stop_strings):
        logger.info(f"Stop token '{next_token_str}' found; terminating early")
        return

      if token_num % 10 == 0:
        gc.collect()
        mx.clear_cache()

      if (token_num + 1) % 10 == 0:
        logger.info(f"Generated {token_num + 1}/{max_tokens} tokens")

    logger.info("Generation complete")

  except Exception as e:
    logger.error(f"Generation failed: {e}")
    raise
  finally:
    if drift_corrector is not None:
      drift_corrector.close()


def generate_with_logit_lens(
  model: Any,
  tokenizer: Any,
  prompt: str,
  max_tokens: int = 100,
  log_from_token: int = 0,
  top_k: int = 5,
  temperature: float = 1.0,
  top_p: float | None = None,
  sampling_method: str = "greedy",
  seed: int | None = None,
  include_prompt: bool = False,
  show_progress: bool = True,
  stop_strings: list[str] | None = None,
) -> dict[str, Any]:
  """Generate tokens and return collected results (backward-compatible wrapper).

  Returns:
    Dict with prompt, generated_tokens, token_predictions, config.
  """
  prompt_ids = tokenizer.encode(prompt)
  if not isinstance(prompt_ids, list):
    prompt_ids = prompt_ids.tolist()

  token_predictions = []
  generated_tokens = []

  for result in stream_with_logit_lens(
    model,
    tokenizer,
    prompt,
    max_tokens=max_tokens,
    log_from_token=log_from_token,
    top_k=top_k,
    temperature=temperature,
    top_p=top_p,
    sampling_method=sampling_method,
    seed=seed,
    stop_strings=stop_strings,
  ):
    generated_tokens.append(result.token_str)
    if result.layer_predictions:
      token_predictions.append({
        "token_number": result.token_num,
        "generated_token": result.token_str,
        "generated_token_id": result.token_id,
        "layer_predictions": result.layer_predictions,
      })

  return {
    "prompt": prompt,
    "prompt_tokens": prompt_ids,
    "generated_tokens": generated_tokens,
    "token_predictions": token_predictions,
    "config": {
      "max_tokens": max_tokens,
      "log_from_token": log_from_token,
      "top_k": top_k,
      "temperature": temperature,
      "top_p": top_p,
      "sampling_method": sampling_method,
      "seed": seed,
      "include_prompt": include_prompt,
      "logged_tokens": len(token_predictions),
    },
  }
