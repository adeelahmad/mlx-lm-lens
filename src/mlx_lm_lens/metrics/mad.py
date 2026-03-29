"""Mean Absolute Deviation (L2 norm difference) metric."""

import numpy as np

from .base import MetricComputationError, register_metric


@register_metric
class MADMetric:
    """L2 norm difference as percentage of RMS."""

    name = "mad"

    @staticmethod
    def compute(activations_a: np.ndarray, activations_b: np.ndarray) -> float:
        """Compute L2 norm difference percentage.

        Args:
            activations_a: (seq_len, hidden_dim) array.
            activations_b: (seq_len, hidden_dim) array.

        Returns:
            float: Percent difference in L2 norm (0-100+).

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

            rms_a = float(np.sqrt(np.mean(activations_a**2)))
            rms_b = float(np.sqrt(np.mean(activations_b**2)))

            if rms_a < 1e-9:
                if rms_b < 1e-9:
                    return 0.0
                return float("inf")

            result = abs(rms_b - rms_a) / rms_a * 100.0
            if not np.isfinite(result):
                raise MetricComputationError(f"Non-finite result: {result}")
            return result
        except MetricComputationError:
            raise
        except Exception as e:
            raise MetricComputationError(f"MAD computation failed: {e}") from e
