"""Effective dimensionality metric via SVD entropy."""

import numpy as np

from .base import MetricComputationError, register_metric


@register_metric
class EffectiveDimMetric:
    """Effective rank via SVD entropy."""

    name = "effective_dim"

    @staticmethod
    def compute(
        activations_a: np.ndarray,
        activations_b: np.ndarray,
        max_rows: int = 500,
    ) -> float:
        """Compute effective dimensionality of concatenated activations.

        Uses SVD entropy: exp(entropy(normalized_singular_values)).
        1.0 = one dominant direction, dim = spread across directions.

        Args:
            activations_a: (seq_len, hidden_dim) array.
            activations_b: (seq_len, hidden_dim) array.
            max_rows: Maximum samples to use per array.

        Returns:
            float: Effective rank (scalar > 0).

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

            # Concatenate along sequence dimension
            n_a = min(activations_a.shape[0], max_rows)
            n_b = min(activations_b.shape[0], max_rows)
            d = min(activations_a.shape[1], activations_b.shape[1])

            a = activations_a[:n_a, :d].astype(np.float64)
            b = activations_b[:n_b, :d].astype(np.float64)
            combined = np.vstack([a, b])

            if combined.shape[0] < 2 or combined.shape[1] < 1:
                raise MetricComputationError("Insufficient samples for SVD")

            # Compute SVD entropy
            sv = np.linalg.svd(combined, compute_uv=False)
            sv = sv[sv > 0]
            if sv.size == 0:
                raise MetricComputationError("No positive singular values")

            p = sv / sv.sum()
            entropy = -float(np.sum(p * np.log(p + 1e-12)))
            result = float(np.exp(entropy))

            if not np.isfinite(result):
                raise MetricComputationError(f"Non-finite result: {result}")
            return result
        except MetricComputationError:
            raise
        except Exception as e:
            raise MetricComputationError(
                f"Effective dimensionality computation failed: {e}"
            ) from e
