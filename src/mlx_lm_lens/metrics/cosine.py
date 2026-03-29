"""Cosine similarity metric."""

import numpy as np

from .base import MetricComputationError, register_metric


@register_metric
class CosineMetric:
    """Flattened cosine similarity between activation matrices."""

    name = "cosine"

    @staticmethod
    def compute(activations_a: np.ndarray, activations_b: np.ndarray) -> float:
        """Compute cosine similarity.

        Args:
            activations_a: (seq_len, hidden_dim) array.
            activations_b: (seq_len, hidden_dim) array.

        Returns:
            float: Cosine similarity in [0, 1] (with numerical tolerance).

        Raises:
            MetricComputationError: If inputs are invalid.
        """
        try:
            if activations_a.size == 0 or activations_b.size == 0:
                raise MetricComputationError("Empty activation arrays")
            if not np.all(np.isfinite(activations_a)):
                raise MetricComputationError("NaN/inf in activations_a")
            if not np.all(np.isfinite(activations_b)):
                raise MetricComputationError("NaN/inf in activations_b")

            va = activations_a.reshape(-1).astype(np.float64)
            vb = activations_b.reshape(-1).astype(np.float64)
            denom = np.linalg.norm(va) * np.linalg.norm(vb) + 1e-12
            result = float(np.dot(va, vb) / denom)

            if not np.isfinite(result):
                raise MetricComputationError(f"Non-finite result: {result}")
            return result
        except MetricComputationError:
            raise
        except Exception as e:
            raise MetricComputationError(f"Cosine computation failed: {e}") from e
