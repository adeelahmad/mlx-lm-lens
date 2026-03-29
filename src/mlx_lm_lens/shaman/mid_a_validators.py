"""SHAMAN validators for H5-H6 (sublayer & sign consistency)."""

from typing import Dict, Optional
import numpy as np
from mlx_lm_lens.shaman.validators import HypothesisResult, _get_layer_keys, _get_valid_values


class MidAValidator:
    """Validators for sublayer and sign consistency hypotheses."""

    @staticmethod
    def validate_h5_sublayer_attn_vs_mlp(
        sublayer_sim: Optional[Dict[str, Dict]],
        num_layers: int,
    ) -> HypothesisResult:
        """H5: Mid layers (10-22): attn vs MLP divergence."""
        if not sublayer_sim:
            return HypothesisResult(
                hypothesis_id="H5",
                name="H5_sublayer_attn_vs_mlp",
                description="Mid: attn vs MLP sublayer divergence",
                passed=None,
                evidence="Sublayer capture not enabled.",
                metrics={},
            )
        keys = [k for k in sublayer_sim if any(f"layer_{i}" == k for i in range(10, 23))]
        if not keys:
            return HypothesisResult(
                hypothesis_id="H5",
                name="H5_sublayer_attn_vs_mlp",
                description="Mid: attn vs MLP sublayer divergence",
                passed=None,
                evidence="No mid-layer sublayer data.",
                metrics={},
            )
        attn_valid = _get_valid_values(keys, sublayer_sim, "attn_cos")
        mlp_valid = _get_valid_values(keys, sublayer_sim, "mlp_cos")
        if not attn_valid or not mlp_valid:
            return HypothesisResult(
                hypothesis_id="H5",
                name="H5_sublayer_attn_vs_mlp",
                description="Mid: attn vs MLP sublayer divergence",
                passed=None,
                evidence="Insufficient sublayer data.",
                metrics={},
            )
        mean_attn = float(np.mean(attn_valid))
        mean_mlp = float(np.mean(mlp_valid))
        divergence_exists = abs(mean_attn - mean_mlp) > 0.0
        return HypothesisResult(
            hypothesis_id="H5",
            name="H5_sublayer_attn_vs_mlp",
            description="Mid: attn vs MLP sublayer divergence",
            passed=True if divergence_exists else None,
            evidence=f"attn_cos={mean_attn:.5f}, mlp_cos={mean_mlp:.5f}",
            metrics={"mean_attn_cos": mean_attn, "mean_mlp_cos": mean_mlp},
        )

    @staticmethod
    def validate_h6_sign_consistency(
        neuron_metrics: Dict[str, Dict],
        num_layers: int,
    ) -> HypothesisResult:
        """H6: Mid layers (10-22): sign consistency ≥ 0.9."""
        keys = _get_layer_keys(10, 22, neuron_metrics)
        if not keys:
            return HypothesisResult(
                hypothesis_id="H6",
                name="H6_sign_consistency",
                description="Mid (10-22): sign consistency ≥0.9",
                passed=None,
                evidence="No neuron metrics.",
                metrics={},
            )
        valid = _get_valid_values(keys, neuron_metrics, "sign_consistency")
        if not valid:
            return HypothesisResult(
                hypothesis_id="H6",
                name="H6_sign_consistency",
                description="Mid (10-22): sign consistency ≥0.9",
                passed=None,
                evidence="No sign consistency data.",
                metrics={},
            )
        mean_val = float(np.mean(valid))
        passed = mean_val >= 0.9
        return HypothesisResult(
            hypothesis_id="H6",
            name="H6_sign_consistency",
            description="Mid (10-22): sign consistency ≥0.9",
            passed=passed,
            evidence=f"mean sign_consistency={mean_val:.4f}",
            metrics={"mean_sign_consistency": mean_val},
        )
