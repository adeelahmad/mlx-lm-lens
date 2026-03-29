"""SHAMAN hypothesis validation runner."""

import logging
from typing import Any, Dict, List, Optional

import numpy as np

from mlx_lm_lens.shaman.hypotheses import HypothesisResult, HypothesisValidator

logger = logging.getLogger(__name__)


class SHAMANValidator:
    """Validate all 15 SHAMAN hypotheses with metrics data."""

    @staticmethod
    def validate_all_hypotheses(
        similarity: Dict[str, Dict[str, float]],
        neuron_metrics: Optional[Dict[str, Dict[str, float]]] = None,
        sublayer_sim: Optional[Dict[str, Dict[str, float]]] = None,
        control_sim: Optional[Dict[str, Dict[str, float]]] = None,
        contrib_angles_a: Optional[Dict[str, float]] = None,
        contrib_angles_b: Optional[Dict[str, float]] = None,
        logit_kl: Optional[Dict[str, float]] = None,
        attn_js: Optional[Dict[str, float]] = None,
        num_layers: int = 36,
    ) -> Dict[str, Any]:
        """Run all 15 hypothesis validators.

        Args:
            similarity: Dict[layer_key] = {metric_name: value}.
            neuron_metrics: Layer neuron activation metrics.
            sublayer_sim: Similarity within layers (attn vs mlp).
            control_sim: Control/baseline similarity.
            contrib_angles_a: Contribution angles for model A.
            contrib_angles_b: Contribution angles for model B.
            logit_kl: KL divergence of logit distributions.
            attn_js: JS divergence of attention patterns.
            num_layers: Total number of layers.

        Returns:
            Dict with:
              - "results": Dict[hypothesis_id] = HypothesisResult
              - "summary": {"passed": int, "failed": int, "na": int}
        """
        neuron_metrics = neuron_metrics or {}
        sublayer_sim = sublayer_sim or {}
        control_sim = control_sim or {}
        contrib_angles_a = contrib_angles_a or {}
        contrib_angles_b = contrib_angles_b or {}
        logit_kl = logit_kl or {}
        attn_js = attn_js or {}

        results: Dict[str, HypothesisResult] = {}
        summary = {"passed": 0, "failed": 0, "na": 0}

        # H1: Early preservation
        h1 = HypothesisValidator.validate_h1_early_preservation(
            similarity, num_layers
        )
        results["H1"] = h1
        _update_summary(h1, summary)

        # H2: Mid rewiring
        h2 = HypothesisValidator.validate_h2_mid_rewiring(
            neuron_metrics, num_layers
        )
        results["H2"] = h2
        _update_summary(h2, summary)

        # H3: Deep geometric transformation
        h3 = HypothesisValidator.validate_h3_deep_geometric(
            similarity, num_layers
        )
        results["H3"] = h3
        _update_summary(h3, summary)

        # H4: Task specificity
        h4 = HypothesisValidator.validate_h4_task_specificity(
            control_sim, similarity
        )
        results["H4"] = h4
        _update_summary(h4, summary)

        # H5: Sublayer attn vs mlp
        h5 = HypothesisValidator.validate_h5_sublayer_attn_vs_mlp(
            sublayer_sim, num_layers
        )
        results["H5"] = h5
        _update_summary(h5, summary)

        # H6: Sign consistency
        h6 = HypothesisValidator.validate_h6_sign_consistency(
            neuron_metrics, num_layers
        )
        results["H6"] = h6
        _update_summary(h6, summary)

        # H7: Magnitude drift
        h7 = HypothesisValidator.validate_h7_magnitude_drift(
            neuron_metrics, num_layers
        )
        results["H7"] = h7
        _update_summary(h7, summary)

        # H8: Angular deviation
        h8 = HypothesisValidator.validate_h8_angular_deviation(
            contrib_angles_a, contrib_angles_b
        )
        results["H8"] = h8
        _update_summary(h8, summary)

        # H9: Procrustes residual
        h9 = HypothesisValidator.validate_h9_procrustes(
            similarity, num_layers
        )
        results["H9"] = h9
        _update_summary(h9, summary)

        # H10: Anisotropy shift
        h10 = HypothesisValidator.validate_h10_anisotropy_shift(
            neuron_metrics, num_layers
        )
        results["H10"] = h10
        _update_summary(h10, summary)

        # H11: Effective rank
        h11 = HypothesisValidator.validate_h11_effective_rank(
            similarity, num_layers
        )
        results["H11"] = h11
        _update_summary(h11, summary)

        # H12: Neuron correlation
        h12 = HypothesisValidator.validate_h12_neuron_correlation(
            neuron_metrics, num_layers
        )
        results["H12"] = h12
        _update_summary(h12, summary)

        # H13: Contribution angle
        h13 = HypothesisValidator.validate_h13_contribution_angle(
            contrib_angles_a, contrib_angles_b
        )
        results["H13"] = h13
        _update_summary(h13, summary)

        # H14: Logit KL peak
        h14 = HypothesisValidator.validate_h14_logit_kl_peak(logit_kl)
        results["H14"] = h14
        _update_summary(h14, summary)

        # H15: Attention JS early
        h15 = HypothesisValidator.validate_h15_attn_js_early(
            attn_js, num_layers
        )
        results["H15"] = h15
        _update_summary(h15, summary)

        logger.info(
            f"SHAMAN validation complete: "
            f"{summary['passed']} passed, "
            f"{summary['failed']} failed, "
            f"{summary['na']} N/A"
        )

        return {
            "results": results,
            "summary": summary,
        }


def _update_summary(
    result: HypothesisResult, summary: Dict[str, int]
) -> None:
    """Update pass/fail/N/A counts."""
    if result.passed is True:
        summary["passed"] += 1
    elif result.passed is False:
        summary["failed"] += 1
    else:
        summary["na"] += 1
