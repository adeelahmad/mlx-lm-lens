"""SHAMAN validators for H7, H11, H13-H15 (deep/specialized)."""

from typing import Dict, Optional
import numpy as np
from mlx_lm_lens.shaman.validators import HypothesisResult, _get_layer_keys, _get_valid_values


class DeepValidator:
    """Validators for deep-layer and specialized hypotheses."""

    @staticmethod
    def validate_h7_magnitude_drift(
        neuron_metrics: Dict[str, Dict],
        num_layers: int,
    ) -> HypothesisResult:
        """H7: Deep layers (23-35): L2 norm diff <5%."""
        keys = _get_layer_keys(23, num_layers - 1, neuron_metrics)
        if not keys:
            return HypothesisResult(
                hypothesis_id="H7",
                name="H7_magnitude_drift",
                description="Deep (23-35): L2 diff <5%",
                passed=None,
                evidence="No deep layer data.",
                metrics={},
            )
        valid = _get_valid_values(keys, neuron_metrics, "l2_diff_pct")
        if not valid:
            return HypothesisResult(
                hypothesis_id="H7",
                name="H7_magnitude_drift",
                description="Deep (23-35): L2 diff <5%",
                passed=None,
                evidence="No L2 data.",
                metrics={},
            )
        mean_val = float(np.mean(valid))
        passed = mean_val < 5.0
        return HypothesisResult(
            hypothesis_id="H7",
            name="H7_magnitude_drift",
            description="Deep (23-35): L2 diff <5%",
            passed=passed,
            evidence=f"deep mean L2_diff={mean_val:.2f}%",
            metrics={"l2_diff_pct": mean_val},
        )

    @staticmethod
    def validate_h11_effective_rank(
        neuron_metrics: Dict[str, Dict],
        num_layers: int,
    ) -> HypothesisResult:
        """H11: Early layers (0-9): effective rank diff <5%."""
        keys = _get_layer_keys(0, 9, neuron_metrics)
        if not keys:
            return HypothesisResult(
                hypothesis_id="H11",
                name="H11_effective_rank",
                description="Early (0-9): eff_rank_diff <5%",
                passed=None,
                evidence="No neuron metrics.",
                metrics={},
            )
        valid = _get_valid_values(keys, neuron_metrics, "effective_rank_diff_pct")
        if not valid:
            return HypothesisResult(
                hypothesis_id="H11",
                name="H11_effective_rank",
                description="Early (0-9): eff_rank_diff <5%",
                passed=None,
                evidence="No effective rank data.",
                metrics={},
            )
        mean_val = float(np.mean(valid))
        passed = mean_val < 5.0
        return HypothesisResult(
            hypothesis_id="H11",
            name="H11_effective_rank",
            description="Early (0-9): eff_rank_diff <5%",
            passed=passed,
            evidence=f"mean eff_rank_diff={mean_val:.2f}%",
            metrics={"eff_rank_diff_pct": mean_val},
        )

    @staticmethod
    def validate_h13_contribution_angle(
        contrib_angles_a: Optional[Dict[str, float]],
        contrib_angles_b: Optional[Dict[str, float]],
        num_layers: int,
    ) -> HypothesisResult:
        """H13: Adapted layers (10-22): contribution angle >75°."""
        if not contrib_angles_b:
            return HypothesisResult(
                hypothesis_id="H13",
                name="H13_contribution_angle",
                description="Adapted (10-22): contrib_angle >75°",
                passed=None,
                evidence="Contribution angle data not available.",
                metrics={},
            )
        adapted_keys = [f"layer_{i}" for i in range(10, 23)]
        ca_vals = [
            contrib_angles_b[k]
            for k in adapted_keys
            if k in contrib_angles_b and np.isfinite(contrib_angles_b[k])
        ]
        if not ca_vals:
            return HypothesisResult(
                hypothesis_id="H13",
                name="H13_contribution_angle",
                description="Adapted (10-22): contrib_angle >75°",
                passed=None,
                evidence="No valid contribution angle data.",
                metrics={},
            )
        mean_ca = float(np.mean(ca_vals))
        passed = mean_ca > 75.0
        return HypothesisResult(
            hypothesis_id="H13",
            name="H13_contribution_angle",
            description="Adapted (10-22): contrib_angle >75°",
            passed=passed,
            evidence=f"mean contribution_angle={mean_ca:.2f}°",
            metrics={"contribution_angle_deg": mean_ca},
        )

    @staticmethod
    def validate_h14_logit_kl_peak(
        logit_kl: Optional[Dict[str, float]],
        num_layers: int,
    ) -> HypothesisResult:
        """H14: Logit KL peaks mid-deep (20-28) > early (0-9) & final (33-35)."""
        if not logit_kl:
            return HypothesisResult(
                hypothesis_id="H14",
                name="H14_logit_kl_peak",
                description="KL peak at mid-deep > early & final",
                passed=None,
                evidence="Logit KL data not available.",
                metrics={},
            )

        def _get_kl_vals(lo: int, hi: int) -> list:
            return [
                logit_kl[f"layer_{i}"]
                for i in range(lo, hi + 1)
                if f"layer_{i}" in logit_kl and np.isfinite(logit_kl[f"layer_{i}"])
            ]

        early_v = _get_kl_vals(0, 9)
        mid_v = _get_kl_vals(20, 28)
        final_v = _get_kl_vals(33, 35)
        if not early_v or not mid_v or not final_v:
            return HypothesisResult(
                hypothesis_id="H14",
                name="H14_logit_kl_peak",
                description="KL peak at mid-deep > early & final",
                passed=None,
                evidence="Insufficient logit KL data.",
                metrics={},
            )
        early_m = float(np.mean(early_v))
        mid_m = float(np.mean(mid_v))
        final_m = float(np.mean(final_v))
        passed = mid_m > early_m and mid_m > final_m
        return HypothesisResult(
            hypothesis_id="H14",
            name="H14_logit_kl_peak",
            description="KL peak at mid-deep > early & final",
            passed=passed,
            evidence=f"KL: early={early_m:.4f}, mid={mid_m:.4f}, final={final_m:.4f}",
            metrics={"early_kl": early_m, "mid_kl": mid_m, "final_kl": final_m},
        )

    @staticmethod
    def validate_h15_attn_js_early(
        attn_js: Optional[Dict[str, float]],
        num_layers: int,
    ) -> HypothesisResult:
        """H15: Early layers (0-9): attention JS divergence <0.05."""
        if not attn_js:
            return HypothesisResult(
                hypothesis_id="H15",
                name="H15_attn_js_early",
                description="Early (0-9): attn JS div <0.05",
                passed=None,
                evidence="Attention JS data not available.",
                metrics={},
            )
        early_vals = [
            attn_js[f"layer_{i}"]
            for i in range(10)
            if f"layer_{i}" in attn_js and np.isfinite(attn_js[f"layer_{i}"])
        ]
        if not early_vals:
            return HypothesisResult(
                hypothesis_id="H15",
                name="H15_attn_js_early",
                description="Early (0-9): attn JS div <0.05",
                passed=None,
                evidence="No valid attention JS data.",
                metrics={},
            )
        mean_js = float(np.mean(early_vals))
        passed = mean_js < 0.05
        return HypothesisResult(
            hypothesis_id="H15",
            name="H15_attn_js_early",
            description="Early (0-9): attn JS div <0.05",
            passed=passed,
            evidence=f"mean attn_js={mean_js:.5f}",
            metrics={"attn_js_divergence": mean_js},
        )
