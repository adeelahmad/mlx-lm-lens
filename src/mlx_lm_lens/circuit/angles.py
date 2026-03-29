"""Weight angle analysis: measure rotation between checkpoints."""

import logging
from pathlib import Path
from typing import Any, Dict, Optional

import mlx.core as mx
import numpy as np
from mlx.utils import tree_flatten

logger = logging.getLogger(__name__)


class WeightAngleAnalyzer:
  """Compute weight angles between two model checkpoints."""

  @staticmethod
  def compute_weight_angles(
    adapter_a_path: str,
    adapter_b_path: str,
    per_layer: bool = False,
  ) -> Dict[str, float]:
    """Compute angles between weight matrices of two adapters.

    For each layer, angle = arccos(|trace(W_A^T W_B)| / (||W_A|| ||W_B||))

    Args:
      adapter_a_path: Path to first adapter checkpoint.
      adapter_b_path: Path to second adapter checkpoint.
      per_layer: If True, return per-layer angles. If False, aggregate.

    Returns:
      Dict with per-layer or aggregated angle values (in degrees).
    """
    logger.debug(f"Loading adapter A: {adapter_a_path}")
    weights_a = _load_adapter_weights(adapter_a_path)

    logger.debug(f"Loading adapter B: {adapter_b_path}")
    weights_b = _load_adapter_weights(adapter_b_path)

    angles = {}

    # Iterate over common weight keys
    common_keys = set(weights_a.keys()) & set(weights_b.keys())

    for key in sorted(common_keys):
      w_a = weights_a[key]
      w_b = weights_b[key]

      if w_a.shape != w_b.shape:
        logger.warning(
          f"Shape mismatch for {key}: {w_a.shape} vs {w_b.shape}"
        )
        continue

      angle = _compute_weight_angle(w_a, w_b)
      angles[key] = angle

    if not per_layer:
      # Return aggregated angle (mean)
      mean_angle = float(np.mean(list(angles.values())))
      return {"aggregate": mean_angle}

    return angles

  @staticmethod
  def compute_cumulative_rotation(
    checkpoint_paths: list[str],
  ) -> Dict[str, float]:
    """Compute cumulative rotation across multiple checkpoints.

    Args:
      checkpoint_paths: List of checkpoint paths in chronological order.

    Returns:
      Dict with cumulative angles between consecutive checkpoints.
    """
    rotations = {}

    for i in range(len(checkpoint_paths) - 1):
      path_a = checkpoint_paths[i]
      path_b = checkpoint_paths[i + 1]

      angles = WeightAngleAnalyzer.compute_weight_angles(
        path_a,
        path_b,
        per_layer=False,
      )

      key = f"checkpoint_{i}_to_{i+1}"
      rotations[key] = angles.get("aggregate", 0.0)

    return rotations


def _load_adapter_weights(adapter_path: str) -> Dict[str, np.ndarray]:
  """Load weights from adapter checkpoint.

  Args:
    adapter_path: Path to adapter or model weights.

  Returns:
    Dict mapping weight name → numpy array.
  """
  path = Path(adapter_path)

  if not path.exists():
    raise FileNotFoundError(f"Adapter path not found: {adapter_path}")

  weights = {}

  # Try to load from MLX safetensors format
  try:
    import mlx.core as mx
    from mlx_lm.utils import save, load

    # Attempt to load using mlx_lm utilities
    state = {}
    if path.suffix == ".safetensors":
      from safetensors.numpy import load_file
      weights_data = load_file(str(path))
      for k, v in weights_data.items():
        weights[k] = v
    else:
      # Assume it's a directory with model files
      logger.debug(f"Loading from directory: {path}")
      # This is a simplified version - real implementation would
      # use mlx_lm.utils.load to handle various formats
      weights = _load_directory_weights(path)

  except Exception as e:
    logger.error(f"Failed to load weights from {adapter_path}: {e}")
    raise

  return weights


def _load_directory_weights(path: Path) -> Dict[str, np.ndarray]:
  """Load weights from directory structure."""
  weights = {}

  # Look for common weight file patterns
  for pattern in ["*.safetensors", "*.npz", "*.npy"]:
    import glob
    for weight_file in glob.glob(str(path / pattern)):
      try:
        if weight_file.endswith(".safetensors"):
          from safetensors.numpy import load_file
          file_weights = load_file(weight_file)
          weights.update(file_weights)
      except Exception as e:
        logger.warning(f"Failed to load {weight_file}: {e}")

  if not weights:
    logger.warning(f"No weights found in {path}")

  return weights


def _compute_weight_angle(
  w_a: np.ndarray,
  w_b: np.ndarray,
) -> float:
  """Compute angle between two weight matrices.

  angle = arccos(|trace(W_A^T W_B)| / (||W_A|| ||W_B||))

  Args:
    w_a: Weight matrix A.
    w_b: Weight matrix B (same shape as w_a).

  Returns:
    Angle in degrees.
  """
  # Flatten weights to vectors
  w_a_flat = w_a.flatten()
  w_b_flat = w_b.flatten()

  # Compute cosine similarity
  dot_product = np.dot(w_a_flat, w_b_flat)
  norm_a = np.linalg.norm(w_a_flat)
  norm_b = np.linalg.norm(w_b_flat)

  if norm_a == 0 or norm_b == 0:
    logger.warning("Zero norm encountered, returning 0 angle")
    return 0.0

  cosine_sim = abs(dot_product) / (norm_a * norm_b)
  cosine_sim = np.clip(cosine_sim, -1.0, 1.0)

  angle_radians = np.arccos(cosine_sim)
  angle_degrees = np.degrees(angle_radians)

  return float(angle_degrees)
