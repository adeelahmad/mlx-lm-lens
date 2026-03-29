"""Procrustes distance metric."""

import math
import numpy as np

from .base import MetricComputationError, register_metric


@register_metric
class ProcustesMetric:
    """Orthogonal Procrustes: best-fit rotation distance."""

    name = "procrustes"

    @staticmethod
    def compute(
        activations_a: np.ndarray,
        activations_b: np.ndarray,
        max_rows: int = 512,
        proj_dim: int = 64,
    ) -> float:
        """Compute Procrustes distance.

        Find rotation R minimizing ||A - BR||_F. Projects to proj_dim first to
        keep SVD tractable.

        Args:
            activations_a: (seq_len, hidden_dim) array.
            activations_b: (seq_len, hidden_dim) array.
            max_rows: Maximum samples to use.
            proj_dim: Project to this dimension before SVD.

        Returns:
            float: Procrustes distance (normalized residual).

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
            idx = np.random.RandomState(0).choice(n_available, n, replace=False)
            d = min(activations_a.shape[1], activations_b.shape[1])

            a = activations_a[idx, :d].astype(np.float64)
            b = activations_b[idx, :d].astype(np.float64)

            # Random projection if needed
            if d > proj_dim:
                P = np.random.RandomState(42).randn(d, proj_dim) / math.sqrt(proj_dim)
                a = a @ P
                b = b @ P

            # Normalize by Frobenius norm
            a = a / (np.linalg.norm(a, "fro") + 1e-12)
            b = b / (np.linalg.norm(b, "fro") + 1e-12)

            # SVD to find best rotation
            U, S, Vt = np.linalg.svd(b.T @ a, full_matrices=False)
            R = U @ Vt
            dist = float(np.linalg.norm(a - b @ R, "fro"))

            if not np.isfinite(dist):
                raise MetricComputationError(f"Non-finite result: {dist}")
            return dist
        except MetricComputationError:
            raise
        except Exception as e:
            raise MetricComputationError(f"Procrustes computation failed: {e}") from e
