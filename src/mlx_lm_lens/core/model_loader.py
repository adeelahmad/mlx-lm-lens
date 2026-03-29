"""Model loading and memory management for MLX models."""

import gc
import logging
from typing import Any, List, Optional, Protocol, Tuple

import mlx.core as mx
from mlx_lm.utils import load

logger = logging.getLogger(__name__)


class MLXModel(Protocol):
  """Abstract interface for loaded MLX models."""

  @property
  def config(self) -> dict:
    """Model configuration dictionary."""
    ...

  @property
  def lm_head(self) -> Optional[Any]:
    """Final unembedding layer (may be None if fused with norm)."""
    ...

  @property
  def norm(self) -> Optional[Any]:
    """Final normalization layer before projection."""
    ...

  @property
  def layers(self) -> List[Any]:
    """Transformer layers."""
    ...

  def __call__(self, x: mx.array) -> mx.array:
    """Forward pass."""
    ...


class ModelLoadError(Exception):
  """Raised when model loading fails."""

  pass


class ModelLoader:
  """Load MLX models with support for quantization and LoRA adapters."""

  @staticmethod
  def load(
    model_path: str,
    adapter_path: Optional[str] = None,
  ) -> Tuple[MLXModel, Any]:
    """Load base model and optionally fuse LoRA adapter.

    Uses mlx_lm.utils.load() which transparently handles quantization
    (4-bit, 8-bit) by loading from adapter weights.

    Args:
      model_path: Path to model directory.
      adapter_path: Optional path to LoRA adapter to fuse.

    Returns:
      Tuple of (model, tokenizer) where model implements MLXModel protocol.

    Raises:
      ModelLoadError: If model or tokenizer cannot be loaded.
    """
    try:
      logger.info(f"Loading model from {model_path}")
      model, tokenizer = load(
        model_path,
        adapter_path=adapter_path,
      )
      logger.info("Model loaded successfully")
      return model, tokenizer
    except FileNotFoundError as e:
      raise ModelLoadError(
        f"Model not found at {model_path}: {e}"
      ) from e
    except Exception as e:
      raise ModelLoadError(f"Failed to load model: {e}") from e

  @staticmethod
  def unload(model: MLXModel) -> None:
    """Free memory allocated by model.

    Cleans up model weights and metal cache on Apple Silicon.

    Args:
      model: Model to unload.
    """
    try:
      del model
      gc.collect()
      mx.clear_cache()
      logger.info("Model unloaded and memory cleared")
    except Exception as e:
      logger.warning(f"Error during cleanup: {e}")
