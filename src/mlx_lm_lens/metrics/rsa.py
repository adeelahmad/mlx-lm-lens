"""Representational Similarity Analysis (RSA) metric."""

import numpy as np

from .base import MetricComputationError, register_metric

try:
    from scipy.stats import spearmanr

    _SCIPY_OK = True
except ImportError:
    _SCIPY_OK = False


def _rsa_distance_vector(activations: np.ndarray, seed: int = 0) -> np.ndarray:
    """Compute upper-triangle distance vector from activations.

    Args:
        activations: (n_samples, hidden_dim) array.
        seed: Random seed for subsampling.

    Returns:
        Upper-triangle cosine distance vector as float32.
    """
    n = activations.shape[0]
    n_subsample = min(1000, n)

    if n > n_subsample:
        acts = activations[
            np.random.RandomState(seed).choice(n, n_subsample, replace=False)
        ]
    else:
        acts = activations

    # Normalize rows and compute cosine distance
    norms = np.linalg.norm(acts, axis=1, keepdims=True) + 1e-12
    normed = acts / norms
    dist = 1.0 - (normed @ normed.T)

    # Extract upper triangle
    iu = np.triu_indices(dist.shape[0], k=1)
    return dist[iu].astype(np.float32)


@register_metric
class RSAMetric:
    """Representational Similarity Analysis via Spearman correlation."""

    name = "rsa"

    @staticmethod
    def compute(activations_a: np.ndarray, activations_b: np.ndarray) -> float:
        """Compute RSA via Spearman correlation of distance vectors.

        CRITICAL BUG FIX: Uses SAME row indices for both models when subsampling
        (not independent samples). This ensures fair comparison of distances
        between the same token pairs.

        Args:
            activations_a: (seq_len, hidden_dim) array.
            activations_b: (seq_len, hidden_dim) array.

        Returns:
            float: Spearman correlation in [-1, 1].

        Raises:
            MetricComputationError: If scipy missing or inputs invalid.
        """
        if not _SCIPY_OK:
            raise MetricComputationError("scipy.stats not available for RSA")

        try:
            if activations_a.size == 0 or activations_b.size == 0:
                raise MetricComputationError("Empty activation arrays")
            if not np.all(np.isfinite(activations_a)):
                raise MetricComputationError("NaN/inf in activations_a")
            if not np.all(np.isfinite(activations_b)):
                raise MetricComputationError("NaN/inf in activations_b")

            # Align sequence length
            n = min(activations_a.shape[0], activations_b.shape[0])
            d = min(activations_a.shape[1], activations_b.shape[1])
            a = activations_a[:n, :d].astype(np.float64)
            b = activations_b[:n, :d].astype(np.float64)

            # Use SAME subsampled indices for both models (critical fix)
            n_subsample = min(1000, n)
            if n > n_subsample:
                indices = np.random.RandomState(42).choice(n, n_subsample, replace=False)
                a = a[indices]
                b = b[indices]

            # Compute distance vectors
            va = _rsa_distance_vector(a, seed=0)
            vb = _rsa_distance_vector(b, seed=0)

            if va.size < 3 or vb.size < 3:
                raise MetricComputationError("Insufficient samples for correlation")

            # Spearman correlation
            r, _ = spearmanr(va, vb)
            result = float(r)

            if not np.isfinite(result):
                raise MetricComputationError(f"Non-finite result: {result}")
            return result
        except MetricComputationError:
            raise
        except Exception as e:
            raise MetricComputationError(f"RSA computation failed: {e}") from e
