"""Metric registry and discovery."""

# Import all metric classes to auto-register them
from . import cosine, cka, mad, procrustes, grassmannian, effective_dim, energy_kl, rsa  # noqa: F401
from .base import get_metric, list_metrics

__all__ = ["get_metric", "list_metrics"]
