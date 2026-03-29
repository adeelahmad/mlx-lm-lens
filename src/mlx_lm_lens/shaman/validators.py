"""SHAMAN hypothesis validators - shared utilities and base structure."""

from dataclasses import dataclass
from typing import Dict, Optional
import numpy as np


@dataclass
class HypothesisResult:
    """Result of hypothesis validation."""

    hypothesis_id: str
    """Hypothesis identifier (e.g., 'H1')."""
    name: str
    """Human-readable name (e.g., 'H1_early_preservation')."""
    description: str
    """The claim being tested."""
    passed: Optional[bool]
    """True/False if testable, None if N/A (insufficient data)."""
    evidence: str
    """Human-readable quantitative explanation."""
    metrics: Dict[str, float]
    """Supporting metric values."""


def _get_layer_keys(lo: int, hi: int, data: Dict) -> list:
    """Extract layer keys in range [lo, hi] from data dict."""
    return [f"layer_{i}" for i in range(lo, hi + 1) if f"layer_{i}" in data]


def _get_valid_values(keys: list, data: Dict, field: str) -> list:
    """Extract valid numeric values from dict entries."""
    values = [data[k].get(field, float("nan")) for k in keys]
    return [v for v in values if np.isfinite(v)]
