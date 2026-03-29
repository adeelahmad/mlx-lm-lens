"""SHAMAN validators for H8-H12 (geometric & correlation)."""

from typing import Dict
import numpy as np
from mlx_lm_lens.shaman.validators import HypothesisResult, _get_layer_keys, _get_valid_values


class MidBValidator:
    """Validators for geometric and correlation hypotheses."""

    @staticmethod
    def validate_h8_angular_deviation(
        neuron_metrics: Dict[str, Dict],
        num_layers: int,
    ) -> HypothesisResult:
        """H8: Mid layers (10-22): MAD <2 degrees."""
        keys = _get_layer_keys(10, 22, neuron_metrics)
        if not keys:
            return HypothesisResult(
                hypothesis_id="H8",
                name="H8_angular_deviation",
                description="Mid (10-22): MAD <2 degrees",
                passed=None,
                evidence="No neuron metrics.",
                metrics={},
            )
        valid = _get_valid_values(keys, neuron_metrics, "mad_mean_deg")
        if not valid:
            return HypothesisResult(
                hypothesis_id="H8",
                name="H8_angular_deviation",
                description="Mid (10-22): MAD <2 degrees",
                passed=None,
                evidence="No MAD data.",
                metrics={},
            )
        mean_val = float(np.mean(valid))
        passed = mean_val < 2.0
        return HypothesisResult(
            hypothesis_id="H8",
            name="H8_angular_deviation",
            description="Mid (10-22): MAD <2 degrees",
            passed=passed,
            evidence=f"mean MAD={mean_val:.3f} degrees",
            metrics={"mean_mad_deg": mean_val},
        )

    @staticmethod
    def validate_h9_procrustes(
        neuron_metrics: Dict[str, Dict],
        num_layers: int,
    ) -> HypothesisResult:
        """H9: Mid layers (10-22): Procrustes distance <0.10."""
        keys = _get_layer_keys(10, 22, neuron_metrics)
        if not keys:
            return HypothesisResult(
                hypothesis_id="H9",
                name="H9_procrustes",
                description="Mid (10-22): Procrustes dist <0.10",
                passed=None,
                evidence="No neuron metrics.",
                metrics={},
            )
        valid = _get_valid_values(keys, neuron_metrics, "procrustes_dist")
        if not valid:
            return HypothesisResult(
                hypothesis_id="H9",
                name="H9_procrustes",
                description="Mid (10-22): Procrustes dist <0.10",
                passed=None,
                evidence="No Procrustes data.",
                metrics={},
            )
        mean_val = float(np.mean(valid))
        passed = mean_val < 0.10
        return HypothesisResult(
            hypothesis_id="H9",
            name="H9_procrustes",
            description="Mid (10-22): Procrustes dist <0.10",
            passed=passed,
            evidence=f"mean Procrustes_dist={mean_val:.4f}",
            metrics={"procrustes_dist": mean_val},
        )

    @staticmethod
    def validate_h10_anisotropy_shift(
        neuron_metrics: Dict[str, Dict],
        num_layers: int,
    ) -> HypothesisResult:
        """H10: Mid layers (10-22): anisotropy B > anisotropy A."""
        keys = _get_layer_keys(10, 22, neuron_metrics)
        if not keys:
            return HypothesisResult(
                hypothesis_id="H10",
                name="H10_anisotropy_shift",
                description="Mid (10-22): anisotropy B > A",
                passed=None,
                evidence="No neuron metrics.",
                metrics={},
            )
        a_valid = _get_valid_values(keys, neuron_metrics, "anisotropy_A")
        b_valid = _get_valid_values(keys, neuron_metrics, "anisotropy_B")
        if not a_valid or not b_valid:
            return HypothesisResult(
                hypothesis_id="H10",
                name="H10_anisotropy_shift",
                description="Mid (10-22): anisotropy B > A",
                passed=None,
                evidence="No anisotropy data.",
                metrics={},
            )
        mean_a = float(np.mean(a_valid))
        mean_b = float(np.mean(b_valid))
        passed = mean_b > mean_a
        return HypothesisResult(
            hypothesis_id="H10",
            name="H10_anisotropy_shift",
            description="Mid (10-22): anisotropy B > A",
            passed=passed,
            evidence=f"anisotropy: {mean_a:.4f} -> {mean_b:.4f}",
            metrics={"anisotropy_a": mean_a, "anisotropy_b": mean_b},
        )

    @staticmethod
    def validate_h12_neuron_correlation(
        neuron_metrics: Dict[str, Dict],
        num_layers: int,
    ) -> HypothesisResult:
        """H12: Mid layers (10-22): neuron correlation >0.85."""
        keys = _get_layer_keys(10, 22, neuron_metrics)
        if not keys:
            return HypothesisResult(
                hypothesis_id="H12",
                name="H12_neuron_correlation",
                description="Mid (10-22): neuron_corr >0.85",
                passed=None,
                evidence="No neuron metrics.",
                metrics={},
            )
        valid = _get_valid_values(keys, neuron_metrics, "neuron_corr")
        if not valid:
            return HypothesisResult(
                hypothesis_id="H12",
                name="H12_neuron_correlation",
                description="Mid (10-22): neuron_corr >0.85",
                passed=None,
                evidence="No neuron correlation data.",
                metrics={},
            )
        mean_val = float(np.mean(valid))
        passed = mean_val > 0.85
        return HypothesisResult(
            hypothesis_id="H12",
            name="H12_neuron_correlation",
            description="Mid (10-22): neuron_corr >0.85",
            passed=passed,
            evidence=f"mean neuron_corr={mean_val:.4f}",
            metrics={"neuron_corr": mean_val},
        )
