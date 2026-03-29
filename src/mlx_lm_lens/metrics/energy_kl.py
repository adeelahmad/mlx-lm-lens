"""Energy KL divergence metric (softmax logit divergence)."""

import numpy as np

from .base import MetricComputationError, register_metric


@register_metric
class EnergyKLMetric:
    """KL divergence of softmax logit distributions."""

    name = "energy_kl"

    @staticmethod
    def compute(activations_a: np.ndarray, activations_b: np.ndarray) -> float:
        """Compute KL divergence of softmax distributions.

        KL(p || q) = sum(p * log(p / (q + eps))) over flattened activations
        after softmax normalization.

        Args:
            activations_a: Logits for model A.
            activations_b: Logits for model B.

        Returns:
            float: KL divergence >= 0.

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

            # Flatten and convert to float64
            p = activations_a.reshape(-1).astype(np.float64)
            q = activations_b.reshape(-1).astype(np.float64)

            # Softmax normalization
            p = p - np.max(p)
            p = np.exp(p)
            p = p / (np.sum(p) + 1e-12)

            q = q - np.max(q)
            q = np.exp(q)
            q = q / (np.sum(q) + 1e-12)

            # KL divergence: sum(p * log(p / q))
            mask = p > 1e-12
            if not np.any(mask):
                raise MetricComputationError("All p values < 1e-12")

            result = float(np.sum(p[mask] * np.log(p[mask] / (q[mask] + 1e-12))))

            if not np.isfinite(result):
                raise MetricComputationError(f"Non-finite result: {result}")
            return result
        except MetricComputationError:
            raise
        except Exception as e:
            raise MetricComputationError(f"Energy KL computation failed: {e}") from e
