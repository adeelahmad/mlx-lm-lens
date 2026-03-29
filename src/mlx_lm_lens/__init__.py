"""mlx-lm-lens: Mechanistic interpretability CLI for MLX language models.

Provides layer-by-layer analysis tools for transformer models running on Apple
Silicon, including logit lens, activation analysis, and circuit discovery.
"""

from mlx_lm_lens.ablations.base import AblationMethod
from mlx_lm_lens.ablations.mean import MeanAblation
from mlx_lm_lens.ablations.zero import ZeroAblation
from mlx_lm_lens.core.activation_capture import ActivationCapture
from mlx_lm_lens.core.logit_generation import LogitLensGenerator
from mlx_lm_lens.core.logit_projection import LogitProjector
from mlx_lm_lens.core.model_loader import MLXModel, ModelLoader
from mlx_lm_lens.core.tokenizer_utils import TokenizerWrapper
from mlx_lm_lens.exceptions import (
    ActivationCaptureError,
    AblationError,
    ConfigError,
    FormattingError,
    MLXLensError,
    MetricComputationError,
    ModelLoadError,
)
from mlx_lm_lens.metrics.base import Metric
from mlx_lm_lens.metrics.cka import CKAMetric
from mlx_lm_lens.metrics.cosine import CosineMetric
from mlx_lm_lens.metrics.effective_dim import EffectiveDimMetric
from mlx_lm_lens.metrics.energy_kl import EnergyKLMetric
from mlx_lm_lens.metrics.grassmannian import GrassmannianMetric
from mlx_lm_lens.metrics.mad import MADMetric
from mlx_lm_lens.metrics.procrustes import ProcustesMetric
from mlx_lm_lens.metrics.rsa import RSAMetric

__version__ = "0.1.0"

__all__ = [
    # Core classes
    "ModelLoader",
    "ActivationCapture",
    "LogitProjector",
    "LogitLensGenerator",
    "TokenizerWrapper",
    # Metrics
    "Metric",
    "CosineMetric",
    "CKAMetric",
    "MADMetric",
    "ProcustesMetric",
    "GrassmannianMetric",
    "EffectiveDimMetric",
    "EnergyKLMetric",
    "RSAMetric",
    # Ablations
    "AblationMethod",
    "ZeroAblation",
    "MeanAblation",
    # Exceptions
    "MLXLensError",
    "ModelLoadError",
    "ActivationCaptureError",
    "MetricComputationError",
    "AblationError",
    "FormattingError",
    "ConfigError",
    # Protocols
    "MLXModel",
]
