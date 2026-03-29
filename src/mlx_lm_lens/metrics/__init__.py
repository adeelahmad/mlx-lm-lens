"""Metric implementations and registry."""

from .base import Metric, MetricComputationError, register_metric
from .cka import CKAMetric
from .cosine import CosineMetric
from .effective_dim import EffectiveDimMetric
from .energy_kl import EnergyKLMetric
from .grassmannian import GrassmannianMetric
from .mad import MADMetric
from .procrustes import ProcustesMetric
from .registry import get_metric, list_metrics
from .rsa import RSAMetric

__all__ = [
    "Metric",
    "MetricComputationError",
    "register_metric",
    "CosineMetric",
    "CKAMetric",
    "MADMetric",
    "ProcustesMetric",
    "GrassmannianMetric",
    "EffectiveDimMetric",
    "EnergyKLMetric",
    "RSAMetric",
    "get_metric",
    "list_metrics",
]
