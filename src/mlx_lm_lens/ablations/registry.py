"""Ablation method registry for plugin discovery."""

from typing import Any, Dict, Type

from mlx_lm_lens.ablations.base import AblationMethod
from mlx_lm_lens.ablations.zero import ZeroAblation
from mlx_lm_lens.ablations.mean import MeanAblation
from mlx_lm_lens.ablations.noise import NoiseAblation
from mlx_lm_lens.ablations.knockout import ResidualKnockout


ABLATION_REGISTRY: Dict[str, Type[AblationMethod]] = {
  "zero": ZeroAblation,
  "mean": MeanAblation,
  "noise": NoiseAblation,
  "knockout": ResidualKnockout,
}


def get_ablation(name: str, **kwargs: Any) -> AblationMethod:
  """Get ablation method by name.

  Args:
      name: Name of ablation method ('zero', 'mean', 'noise', 'knockout').
      **kwargs: Optional parameters for ablation constructor.

  Returns:
      Instantiated ablation method.

  Raises:
      ValueError: If ablation method not found.
  """
  if name not in ABLATION_REGISTRY:
    available = ", ".join(ABLATION_REGISTRY.keys())
    raise ValueError(
      f"Unknown ablation method '{name}'. Available: {available}"
    )

  ablation_class = ABLATION_REGISTRY[name]
  return ablation_class(**kwargs)
