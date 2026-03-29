"""Detect anomalies in activation metrics using rolling statistics."""

import logging
from typing import Dict, List

import numpy as np

logger = logging.getLogger(__name__)


class AnomalyDetector:
    """Detect anomalous layers based on metric deviations."""

    @staticmethod
    def detect_layer_anomalies(
        metrics: Dict[str, Dict[str, float]],
        window_size: int = 3,
        std_threshold: float = 2.0,
    ) -> Dict[str, List[str]]:
        """Detect layers with anomalous metrics.

        Uses rolling window to compute local mean/std, flags layers where
        metric value deviates >std_threshold stds from local neighbors.

        Args:
            metrics: Dict[layer_key][metric_name] = float.
            window_size: Rolling window size (default 3).
            std_threshold: Z-score threshold for flagging (default 2.0).

        Returns:
            Dict[layer_key] = [list of anomalous metric names].
            Empty list if no anomalies detected.
        """
        try:
            anomalies: Dict[str, List[str]] = {}

            # Extract layer indices and sort
            layer_keys = sorted(
                metrics.keys(),
                key=lambda k: _extract_layer_idx(k),
            )

            if not layer_keys:
                return {}

            # For each metric name
            metric_names = set()
            for metrics_dict in metrics.values():
                metric_names.update(metrics_dict.keys())

            for metric_name in metric_names:
                # Extract values for this metric across layers
                values = []
                valid_keys = []
                for key in layer_keys:
                    val = metrics[key].get(metric_name, float("nan"))
                    if np.isfinite(val):
                        values.append(val)
                        valid_keys.append(key)

                if len(values) < 2:
                    continue

                # Compute rolling mean/std
                rolling_means = _rolling_mean(
                    np.array(values), window_size
                )
                rolling_stds = _rolling_std(
                    np.array(values), window_size
                )

                # Flag anomalies
                for i, key in enumerate(valid_keys):
                    if rolling_stds[i] > 0:
                        z_score = abs(
                            (values[i] - rolling_means[i])
                            / rolling_stds[i]
                        )
                        if z_score > std_threshold:
                            if key not in anomalies:
                                anomalies[key] = []
                            anomalies[key].append(metric_name)

            logger.info(
                f"Detected anomalies in {len(anomalies)} layers"
            )
            return anomalies

        except Exception as e:
            logger.error(f"Anomaly detection failed: {e}")
            raise


def _extract_layer_idx(layer_key: str) -> int:
    """Extract numeric layer index from key like 'layer_5'."""
    try:
        return int(layer_key.split("_")[-1])
    except (ValueError, IndexError):
        return 999


def _rolling_mean(arr: np.ndarray, window_size: int) -> np.ndarray:
    """Compute rolling mean with same-size output."""
    if len(arr) < window_size:
        return np.full_like(arr, np.mean(arr), dtype=float)

    means = []
    for i in range(len(arr)):
        start = max(0, i - window_size // 2)
        end = min(len(arr), i + window_size // 2 + 1)
        means.append(np.mean(arr[start:end]))
    return np.array(means)


def _rolling_std(arr: np.ndarray, window_size: int) -> np.ndarray:
    """Compute rolling std with same-size output."""
    if len(arr) < window_size:
        return np.full_like(arr, np.std(arr), dtype=float)

    stds = []
    for i in range(len(arr)):
        start = max(0, i - window_size // 2)
        end = min(len(arr), i + window_size // 2 + 1)
        stds.append(np.std(arr[start:end]))
    return np.array(stds)
