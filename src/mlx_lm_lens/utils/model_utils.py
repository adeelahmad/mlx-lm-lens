"""Model introspection utilities for layer navigation and forward passes."""

from typing import Any, Callable, Optional

import mlx.core as mx
import mlx.nn as nn


def _get_base_model(model: Any) -> Any:
  """Extract base model from wrapper if present.

  Args:
    model: Model or wrapper containing a model.

  Returns:
    Base model with a layers property.
  """
  return (
    model.model
    if (hasattr(model, "model") and hasattr(model.model, "layers"))
    else model
  )


def _get_embed_func(base_model: Any) -> Optional[Callable]:
  """Find embedding function in model.

  Args:
    base_model: Base model to search.

  Returns:
    Embedding function or None.
  """
  for attr in ("embed_tokens", "tok_embeddings", "wte", "embedding"):
    obj = getattr(base_model, attr, None)
    if obj is not None and callable(obj):
      return obj
  return None


def _get_attn_module(layer: Any) -> Optional[Any]:
  """Extract attention module from layer.

  Args:
    layer: Transformer layer.

  Returns:
    Attention module or None.
  """
  return getattr(layer, "self_attn", None) or getattr(
    layer, "attention", None
  )


def _get_mlp_module(layer: Any) -> Optional[Any]:
  """Extract MLP module from layer.

  Args:
    layer: Transformer layer.

  Returns:
    MLP module or None.
  """
  return getattr(layer, "mlp", None) or getattr(
    layer, "feed_forward", None
  )


def _get_norm1(layer: Any) -> Optional[Any]:
  """Extract first normalization module from layer.

  Args:
    layer: Transformer layer.

  Returns:
    Normalization module or None.
  """
  return getattr(layer, "input_layernorm", None) or getattr(
    layer, "ln_1", None
  )


def _get_norm2(layer: Any) -> Optional[Any]:
  """Extract second normalization module from layer.

  Args:
    layer: Transformer layer.

  Returns:
    Normalization module or None.
  """
  return getattr(layer, "post_attention_layernorm", None) or getattr(
    layer, "ln_2", None
  )


def _create_causal_mask(seq_len: int, dtype: Optional[Any] = None) -> mx.array:
  """Create additive causal attention mask for autoregressive models.

  Args:
    seq_len: Sequence length.
    dtype: Optional dtype for mask. Defaults to input dtype.

  Returns:
    Causal attention mask of shape (seq_len, seq_len).
  """
  mask = nn.MultiHeadAttention.create_additive_causal_mask(seq_len)
  if dtype is not None:
    mask = mask.astype(dtype)
  return mask


def _layer_forward(
  layer: Any,
  h: mx.array,
  mask: Optional[mx.array] = None,
) -> mx.array:
  """Forward pass through transformer layer with adaptive signature handling.

  Attempts multiple calling conventions as different models use different
  signatures. Always evaluates result to prevent MLX graph explosion.

  Args:
    layer: Transformer layer module.
    h: Hidden states (batch, seq, hidden) or (seq, hidden).
    mask: Causal attention mask (seq, seq). Auto-created if None.

  Returns:
    Layer output with same shape as input.

  Raises:
    RuntimeError: If all calling strategies fail.
  """
  # Auto-create causal mask if needed
  if mask is None and h.ndim == 3:
    mask = _create_causal_mask(h.shape[1], h.dtype)

  # Strategy 1: mask + cache kwargs (most modern mlx-lm layers)
  try:
    out = layer(h, mask=mask, cache=None)
    result = out[0] if isinstance(out, tuple) else out
    mx.eval(result)
    return result
  except (TypeError, AttributeError):
    pass

  # Strategy 2: mask kwarg only
  try:
    out = layer(h, mask=mask)
    result = out[0] if isinstance(out, tuple) else out
    mx.eval(result)
    return result
  except (TypeError, AttributeError):
    pass

  # Strategy 3: positional (h, mask, cache)
  try:
    out = layer(h, mask, None)
    result = out[0] if isinstance(out, tuple) else out
    mx.eval(result)
    return result
  except (TypeError, AttributeError):
    pass

  # Strategy 4: bare call (some models create mask internally)
  try:
    out = layer(h)
    result = out[0] if isinstance(out, tuple) else out
    mx.eval(result)
    return result
  except Exception as e:
    raise RuntimeError(f"Cannot forward through layer: {e}")
