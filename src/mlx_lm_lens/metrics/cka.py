"""Linear Centered Kernel Alignment (CKA) metric."""

import math
import numpy as np

from .base import MetricComputationError, register_metric


@register_metric
class CKAMetric:
    """Linear Centered Kernel Alignment between activation matrices."""

    name = "cka"

    @staticmethod
    def compute(activations_a: np.ndarray, activations_b: np.ndarray) -> float:
        """Compute linear CKA.

        Args:
            activations_a: (seq_len, hidden_dim) array.
            activations_b: (seq_len, hidden_dim) array.

        Returns:
            float: CKA similarity in [0, 1].

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

            # Center arrays
            Xc = activations_a.astype(np.float64) - activations_a.mean(axis=0, keepdims=True)
            Yc = activations_b.astype(np.float64) - activations_b.mean(axis=0, keepdims=True)

            # Compute HSIC and variances
            hsic = np.linalg.norm(Xc.T @ Yc, "fro") ** 2
            var1 = np.linalg.norm(Xc.T @ Xc, "fro") ** 2
            var2 = np.linalg.norm(Yc.T @ Yc, "fro") ** 2

            denom = math.sqrt(var1) * math.sqrt(var2) + 1e-12
            result = float(hsic / denom)

            if not np.isfinite(result):
                raise MetricComputationError(f"Non-finite result: {result}")
            return result
        except MetricComputationError:
            raise
        except Exception as e:
            raise MetricComputationError(f"CKA computation failed: {e}") from e
