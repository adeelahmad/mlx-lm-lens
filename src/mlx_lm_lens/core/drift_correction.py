"""Geometric drift correction for transformer activations."""

import json
import logging
import math
from typing import Any

import mlx.core as mx

from mlx_lm_lens.exceptions import DriftCorrectionError

logger = logging.getLogger(__name__)


class DriftCorrector:
  """Monitor and correct per-layer activation angular drift during inference."""

  def __init__(
      self,
      threshold_deg: float = 0.3,
      baseline_tokens: int = 256,
      log_path: str | None = None,
  ) -> None:
    """Initialize drift corrector.

    Args:
      threshold_deg: Angle threshold in degrees above which to apply correction.
      baseline_tokens: Number of tokens to use for baseline accumulation.
      log_path: Optional JSONL file path for logging per-layer angles.
    """
    self._threshold_deg = threshold_deg
    self._baseline_tokens = baseline_tokens
    self._log_path = log_path
    self._log_file: Any = None

    self._baselines: dict[int, mx.array] = {}
    self._accumulation_buffer: dict[int, list[mx.array]] = {}
    self._phase = "accumulating"
    self._token_position = 0
    self._last_angles: dict[int, float] = {}

  @property
  def phase(self) -> str:
    """Current phase: 'accumulating' or 'correcting'."""
    return self._phase

  def advance_token(self) -> None:
    """Advance token counter and finalize baselines when threshold is reached."""
    if self._phase == "accumulating" and self._token_position == self._baseline_tokens:
      self._finalize_baselines()
    self._token_position += 1

  def apply(self, layer_idx: int, hidden: mx.array) -> mx.array:
    """Apply drift correction to a layer's hidden state.

    Args:
      layer_idx: Layer index (0-based).
      hidden: Hidden state from layer, shape (1, seq_len, hidden_dim).

    Returns:
      Hidden state (possibly corrected) with original magnitude preserved.
    """
    try:
      if self._phase == "accumulating":
        return self._accumulate(layer_idx, hidden)
      else:
        return self._correct(layer_idx, hidden)
    except Exception as e:
      raise DriftCorrectionError(f"Drift correction failed at layer {layer_idx}: {e}") from e

  def _accumulate(self, layer_idx: int, hidden: mx.array) -> mx.array:
    """Accumulate baseline direction from hidden state."""
    if hidden.ndim == 3:
      h_last = hidden[0, -1, :]
    else:
      h_last = hidden[-1, :]

    h_norm = mx.sqrt(mx.sum(h_last * h_last))
    if float(h_norm) < 1e-8:
      return hidden

    h_unit = h_last / (h_norm + 1e-8)
    mx.eval(h_unit)

    if layer_idx not in self._accumulation_buffer:
      self._accumulation_buffer[layer_idx] = []
    self._accumulation_buffer[layer_idx].append(h_unit)

    return hidden

  def _finalize_baselines(self) -> None:
    """Compute mean baseline direction per layer from accumulated vectors."""
    logger.info(f"Finalizing baselines from {self._baseline_tokens} tokens")

    for layer_idx, buffer in self._accumulation_buffer.items():
      if not buffer:
        continue

      stacked = mx.stack(buffer)
      mean_dir = mx.mean(stacked, axis=0)
      mean_norm = mx.sqrt(mx.sum(mean_dir * mean_dir))

      if float(mean_norm) < 1e-8:
        logger.warning(f"Layer {layer_idx}: baseline direction has zero norm, skipping")
        continue

      baseline = mean_dir / (mean_norm + 1e-8)
      mx.eval(baseline)
      self._baselines[layer_idx] = baseline

    self._accumulation_buffer.clear()
    self._phase = "correcting"
    logger.info(f"Baselines finalized for {len(self._baselines)} layers")

  def _correct(self, layer_idx: int, hidden: mx.array) -> mx.array:
    """Apply rotational correction if drift exceeds threshold."""
    if layer_idx not in self._baselines:
      self._last_angles[layer_idx] = 0.0
      return hidden

    baseline = self._baselines[layer_idx]

    if hidden.ndim == 3:
      h_last = hidden[0, -1, :]
    else:
      h_last = hidden[-1, :]

    mag = mx.sqrt(mx.sum(h_last * h_last))
    if float(mag) < 1e-8:
      self._last_angles[layer_idx] = 0.0
      return hidden

    h_unit = h_last / (mag + 1e-8)
    cos_theta = mx.sum(h_unit * baseline)
    cos_theta_clipped = mx.clip(cos_theta, -1.0, 1.0)
    theta_rad = mx.arccos(cos_theta_clipped)
    theta_deg = float(theta_rad) * (180.0 / math.pi)

    self._last_angles[layer_idx] = theta_deg

    corrected = False
    alpha = None

    if theta_deg > self._threshold_deg:
      corrected = True
      alpha = (theta_deg - self._threshold_deg) / theta_deg

      mag_full = mx.sqrt(mx.sum(hidden * hidden, axis=-1, keepdims=True))
      h_unit_full = hidden / (mag_full + 1e-8)
      baseline_broadcast = baseline

      d = (1.0 - alpha) * h_unit_full + alpha * baseline_broadcast
      d_norm = mx.sqrt(mx.sum(d * d, axis=-1, keepdims=True))
      d_unit = d / (d_norm + 1e-8)
      hidden = d_unit * mag_full
      mx.eval(hidden)

    if self._log_path:
      self._log_angle(layer_idx, theta_deg, corrected, alpha)

    return hidden

  def _log_angle(
      self,
      layer_idx: int,
      angle_deg: float,
      corrected: bool,
      alpha: float | None,
  ) -> None:
    """Log angle measurement to JSONL file."""
    if self._log_file is None:
      try:
        self._log_file = open(self._log_path, "w")
      except OSError as e:
        logger.warning(f"Could not open drift log file: {e}")
        return

    record = {
      "token_pos": self._token_position,
      "layer": layer_idx,
      "angle_deg": round(angle_deg, 3),
      "threshold_deg": self._threshold_deg,
      "corrected": corrected,
      "alpha": round(float(alpha), 3) if alpha is not None else None,
    }

    try:
      self._log_file.write(json.dumps(record) + "\n")
      self._log_file.flush()
    except OSError as e:
      logger.warning(f"Error writing to drift log: {e}")

  def get_last_angles(self) -> dict[int, float]:
    """Get per-layer angles computed during most recent token.

    Returns:
      Dict mapping layer_idx to angle in degrees.
    """
    return self._last_angles.copy()

  def set_reference_baselines(self, baselines: dict[int, mx.array]) -> None:
    """Inject pre-computed reference baselines and skip to correcting phase.

    Args:
      baselines: Dict mapping layer_idx to normalized baseline vectors.
    """
    self._baselines = baselines
    self._accumulation_buffer.clear()
    self._phase = "correcting"
    logger.info(f"Reference baselines set for {len(baselines)} layers")

  def close(self) -> None:
    """Close the JSONL log file if open."""
    if self._log_file is not None:
      try:
        self._log_file.close()
      except OSError as e:
        logger.warning(f"Error closing drift log file: {e}")
