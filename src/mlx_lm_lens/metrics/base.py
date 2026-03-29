"""Base metric interface and registration."""

from typing import Dict, Protocol, Type

import numpy as np


class Metric(Protocol):
    """Interface for representational similarity metrics."""

    name: str

    def compute(
        self,
        activations_a: np.ndarray,
        activations_b: np.ndarray,
    ) -> float:
        """Compute metric between two activation matrices.

        Args:
            activations_a: (seq_len, hidden_dim) array from model A.
            activations_b: (seq_len, hidden_dim) array from model B.

        Returns:
            float: metric value.
        """
        ...


class MetricComputationError(Exception):
    """Raised when metric computation fails."""

    pass


_METRIC_REGISTRY: Dict[str, Type] = {}


def register_metric(cls: Type) -> Type:
    """Decorator to register a metric class.

    Args:
        cls: Metric class with 'name' attribute and 'compute' method.

    Returns:
        The metric class (unchanged).
    """
    name = getattr(cls, "name", None)
    if not name:
        raise ValueError(f"Metric class {cls.__name__} must define 'name' attribute")
    _METRIC_REGISTRY[name] = cls
    return cls


def get_metric(name: str) -> Metric:
    """Get metric instance by name.

    Args:
        name: Metric name (e.g., 'cosine', 'cka').

    Returns:
        Metric instance.

    Raises:
        KeyError: If metric name not found.
    """
    if name not in _METRIC_REGISTRY:
        available = ", ".join(sorted(_METRIC_REGISTRY.keys()))
        raise KeyError(f"Unknown metric '{name}'. Available: {available}")
    return _METRIC_REGISTRY[name]()


def list_metrics() -> list[str]:
    """List all registered metric names.

    Returns:
        Sorted list of metric names.
    """
    return sorted(_METRIC_REGISTRY.keys())
