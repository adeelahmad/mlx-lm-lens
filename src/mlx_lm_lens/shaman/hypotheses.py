"""SHAMAN hypothesis validators (H1-H15) for mechanistic interpretability."""

from typing import Dict, Optional
from mlx_lm_lens.shaman.validators import HypothesisResult
from mlx_lm_lens.shaman.early_validators import EarlyValidator
from mlx_lm_lens.shaman.mid_a_validators import MidAValidator
from mlx_lm_lens.shaman.mid_b_validators import MidBValidator
from mlx_lm_lens.shaman.deep_validators import DeepValidator


class HypothesisValidator:
    """Master validator class composing all H1-H15 validators."""

    # Early validators (H1-H4)
    validate_h1_early_preservation = staticmethod(
        EarlyValidator.validate_h1_early_preservation
    )
    validate_h2_mid_rewiring = staticmethod(EarlyValidator.validate_h2_mid_rewiring)
    validate_h3_deep_geometric = staticmethod(EarlyValidator.validate_h3_deep_geometric)
    validate_h4_task_specificity = staticmethod(
        EarlyValidator.validate_h4_task_specificity
    )

    # Mid validators (H5-H6, H8-H10, H12)
    validate_h5_sublayer_attn_vs_mlp = staticmethod(
        MidAValidator.validate_h5_sublayer_attn_vs_mlp
    )
    validate_h6_sign_consistency = staticmethod(
        MidAValidator.validate_h6_sign_consistency
    )
    validate_h8_angular_deviation = staticmethod(
        MidBValidator.validate_h8_angular_deviation
    )
    validate_h9_procrustes = staticmethod(MidBValidator.validate_h9_procrustes)
    validate_h10_anisotropy_shift = staticmethod(
        MidBValidator.validate_h10_anisotropy_shift
    )
    validate_h12_neuron_correlation = staticmethod(
        MidBValidator.validate_h12_neuron_correlation
    )

    # Deep validators (H7, H11, H13-H15)
    validate_h7_magnitude_drift = staticmethod(
        DeepValidator.validate_h7_magnitude_drift
    )
    validate_h11_effective_rank = staticmethod(
        DeepValidator.validate_h11_effective_rank
    )
    validate_h13_contribution_angle = staticmethod(
        DeepValidator.validate_h13_contribution_angle
    )
    validate_h14_logit_kl_peak = staticmethod(DeepValidator.validate_h14_logit_kl_peak)
    validate_h15_attn_js_early = staticmethod(DeepValidator.validate_h15_attn_js_early)

    @staticmethod
    def validate_all(
        similarity: Dict[str, Dict],
        neuron_metrics: Dict[str, Dict],
        sublayer_sim: Optional[Dict[str, Dict]] = None,
        control_similarity: Optional[Dict[str, Dict]] = None,
        contrib_angles_a: Optional[Dict[str, float]] = None,
        contrib_angles_b: Optional[Dict[str, float]] = None,
        logit_kl: Optional[Dict[str, float]] = None,
        attn_js: Optional[Dict[str, float]] = None,
        num_layers: int = 36,
    ) -> list:
        """Validate all 15 hypotheses.

        Args:
            similarity: Layer-wise cosine/CKA similarity scores.
            neuron_metrics: Per-layer neuron metrics.
            sublayer_sim: Optional sublayer similarity (attn vs MLP).
            control_similarity: Optional control prompt similarity.
            contrib_angles_a: Optional contribution angles for model A.
            contrib_angles_b: Optional contribution angles for model B.
            logit_kl: Optional logit KL divergences.
            attn_js: Optional attention JS divergences.
            num_layers: Total layers in model.

        Returns:
            List of 15 HypothesisResult objects (H1-H15).
        """
        results = [
            HypothesisValidator.validate_h1_early_preservation(similarity, num_layers),
            HypothesisValidator.validate_h2_mid_rewiring(neuron_metrics, num_layers),
            HypothesisValidator.validate_h3_deep_geometric(similarity, num_layers),
            HypothesisValidator.validate_h4_task_specificity(control_similarity, num_layers),
            HypothesisValidator.validate_h5_sublayer_attn_vs_mlp(sublayer_sim, num_layers),
            HypothesisValidator.validate_h6_sign_consistency(neuron_metrics, num_layers),
            HypothesisValidator.validate_h7_magnitude_drift(neuron_metrics, num_layers),
            HypothesisValidator.validate_h8_angular_deviation(neuron_metrics, num_layers),
            HypothesisValidator.validate_h9_procrustes(neuron_metrics, num_layers),
            HypothesisValidator.validate_h10_anisotropy_shift(neuron_metrics, num_layers),
            HypothesisValidator.validate_h11_effective_rank(neuron_metrics, num_layers),
            HypothesisValidator.validate_h12_neuron_correlation(neuron_metrics, num_layers),
            HypothesisValidator.validate_h13_contribution_angle(
                contrib_angles_a, contrib_angles_b, num_layers
            ),
            HypothesisValidator.validate_h14_logit_kl_peak(logit_kl, num_layers),
            HypothesisValidator.validate_h15_attn_js_early(attn_js, num_layers),
        ]
        return results


__all__ = ["HypothesisResult", "HypothesisValidator"]
