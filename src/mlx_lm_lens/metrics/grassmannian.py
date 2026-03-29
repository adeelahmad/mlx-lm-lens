"""Grassmannian (principal angles) metric."""

import numpy as np

from .base import MetricComputationError, register_metric


@register_metric
class GrassmannianMetric:
    """Principal angles between top-k subspaces."""

    name = "grassmannian"

    @staticmethod
    def compute(
        activations_a: np.ndarray,
        activations_b: np.ndarray,
        k: int = 8,
        max_rows: int = 500,
    ) -> float:
        """Compute Grassmannian distance.

        Principal angles between the top-k left-singular subspaces of A and B.
        Geodesic distance sqrt(sum theta_i^2) measures subspace divergence.

        Args:
            activations_a: (seq_len, hidden_dim) array.
            activations_b: (seq_len, hidden_dim) array.
            k: Number of top singular vectors to use.
            max_rows: Maximum samples to use.

        Returns:
            float: Grassmannian distance.

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

            # Subsample and align
            n = min(activations_a.shape[0], activations_b.shape[0], max_rows)
            n_available = min(activations_a.shape[0], activations_b.shape[0])
            idx = np.random.RandomState(1).choice(n_available, n, replace=False)
            d = min(activations_a.shape[1], activations_b.shape[1])

            a = activations_a[idx, :d].astype(np.float64)
            b = activations_b[idx, :d].astype(np.float64)

            k = min(k, d, n - 1)
            if k < 1:
                raise MetricComputationError(f"Cannot compute with k={k}")

            # Get top-k left singular vectors
            Ua = np.linalg.svd(a, full_matrices=False)[0][:, :k]
            Ub = np.linalg.svd(b, full_matrices=False)[0][:, :k]

            # Singular values of Ua.T @ Ub are cosines of principal angles
            sv = np.clip(np.linalg.svd(Ua.T @ Ub, compute_uv=False)[:k], -1.0, 1.0)
            angles = np.degrees(np.arccos(sv))
            dist = float(np.sqrt(np.sum(angles**2)))

            if not np.isfinite(dist):
                raise MetricComputationError(f"Non-finite result: {dist}")
            return dist
        except MetricComputationError:
            raise
        except Exception as e:
            raise MetricComputationError(f"Grassmannian computation failed: {e}") from e
