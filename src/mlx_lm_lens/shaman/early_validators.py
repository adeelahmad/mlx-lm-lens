"""SHAMAN validators for H1-H4 (early & control)."""

from typing import Dict, Optional
import numpy as np
from mlx_lm_lens.shaman.validators import HypothesisResult, _get_layer_keys, _get_valid_values


class EarlyValidator:
    """Validators for early layer and control hypotheses."""

    @staticmethod
    def validate_h1_early_preservation(
        similarity: Dict[str, Dict],
        num_layers: int,
    ) -> HypothesisResult:
        """H1: Early layers (0-9): cos ≥ 0.9999, CKA ≥ 0.9999."""
        keys = _get_layer_keys(0, 9, similarity)
        if not keys:
            return HypothesisResult(
                hypothesis_id="H1",
                name="H1_early_preservation",
                description="Early layers (0-9): cos≥0.9999 AND CKA≥0.9999",
                passed=None,
                evidence="No similarity data available.",
                metrics={},
            )
        cos_valid = _get_valid_values(keys, similarity, "cosine")
        cka_valid = _get_valid_values(keys, similarity, "cka")
        if not cos_valid or not cka_valid:
            return HypothesisResult(
                hypothesis_id="H1",
                name="H1_early_preservation",
                description="Early layers (0-9): cos≥0.9999 AND CKA≥0.9999",
                passed=None,
                evidence="Insufficient valid data.",
                metrics={},
            )
        mean_cos = float(np.mean(cos_valid))
        mean_cka = float(np.mean(cka_valid))
        passed = mean_cos >= 0.9999 and mean_cka >= 0.9999
        return HypothesisResult(
            hypothesis_id="H1",
            name="H1_early_preservation",
            description="Early layers (0-9): cos≥0.9999 AND CKA≥0.9999",
            passed=passed,
            evidence=f"mean_cosine={mean_cos:.5f}, mean_cka={mean_cka:.5f}",
            metrics={"mean_cosine": mean_cos, "mean_cka": mean_cka},
        )

    @staticmethod
    def validate_h2_mid_rewiring(
        neuron_metrics: Dict[str, Dict],
        num_layers: int,
    ) -> HypothesisResult:
        """H2: Mid layers (10-22): sparsity diff 7-30%, L2 diff <5%."""
        keys = _get_layer_keys(10, 22, neuron_metrics)
        if not keys:
            return HypothesisResult(
                hypothesis_id="H2",
                name="H2_mid_rewiring",
                description="Mid layers (10-22): sparsity diff 7-30% AND L2 diff <5%",
                passed=None,
                evidence="No neuron metrics available.",
                metrics={},
            )
        sp_valid = _get_valid_values(keys, neuron_metrics, "sparsity_diff_pct")
        l2_valid = _get_valid_values(keys, neuron_metrics, "l2_diff_pct")
        if not sp_valid or not l2_valid:
            return HypothesisResult(
                hypothesis_id="H2",
                name="H2_mid_rewiring",
                description="Mid layers (10-22): sparsity diff 7-30% AND L2 diff <5%",
                passed=None,
                evidence="Insufficient valid data.",
                metrics={},
            )
        sp_in_range = all(7 <= s <= 30 for s in sp_valid)
        l2_ok = float(np.mean(l2_valid)) < 5.0
        passed = sp_in_range and l2_ok
        return HypothesisResult(
            hypothesis_id="H2",
            name="H2_mid_rewiring",
            description="Mid layers (10-22): sparsity diff 7-30% AND L2 diff <5%",
            passed=passed,
            evidence=f"sparsity_diff={np.mean(sp_valid):.1f}%, L2_diff={np.mean(l2_valid):.2f}%",
            metrics={
                "sparsity_diff_pct": float(np.mean(sp_valid)),
                "l2_diff_pct": float(np.mean(l2_valid)),
            },
        )

    @staticmethod
    def validate_h3_deep_geometric(
        similarity: Dict[str, Dict],
        num_layers: int,
    ) -> HypothesisResult:
        """H3: Deep layers (23-35): ≥50% layers have CKA < cosine."""
        keys = _get_layer_keys(23, min(35, num_layers - 1), similarity)
        if not keys:
            return HypothesisResult(
                hypothesis_id="H3",
                name="H3_deep_geometric",
                description="Deep layers (23-35): ≥50% have CKA < cosine",
                passed=None,
                evidence="No deep layer data.",
                metrics={},
            )
        cka_lt_cos = [
            similarity[k].get("cka", float("nan"))
            < similarity[k].get("cosine", float("nan"))
            for k in keys
        ]
        valid = [v for v in cka_lt_cos if isinstance(v, bool)]
        if not valid:
            return HypothesisResult(
                hypothesis_id="H3",
                name="H3_deep_geometric",
                description="Deep layers (23-35): ≥50% have CKA < cosine",
                passed=None,
                evidence="Insufficient valid data.",
                metrics={},
            )
        pct = float(np.mean(valid)) * 100
        passed = pct >= 50.0
        return HypothesisResult(
            hypothesis_id="H3",
            name="H3_deep_geometric",
            description="Deep layers (23-35): ≥50% have CKA < cosine",
            passed=passed,
            evidence=f"{sum(valid)}/{len(valid)} layers CKA<cosine ({pct:.1f}%)",
            metrics={"pct_cka_lt_cosine": pct},
        )

    @staticmethod
    def validate_h4_task_specificity(
        control_similarity: Optional[Dict[str, Dict]],
        num_layers: int,
    ) -> HypothesisResult:
        """H4: Control prompt: cos ≥ 0.9998, CKA ≥ 0.9998 everywhere."""
        if not control_similarity:
            return HypothesisResult(
                hypothesis_id="H4",
                name="H4_task_specificity",
                description="Control: cos≥0.9998 AND CKA≥0.9998",
                passed=None,
                evidence="Control prompt data not available.",
                metrics={},
            )
        cos_vals = [v.get("cosine", float("nan")) for v in control_similarity.values()]
        cka_vals = [v.get("cka", float("nan")) for v in control_similarity.values()]
        cos_valid = [v for v in cos_vals if np.isfinite(v)]
        cka_valid = [v for v in cka_vals if np.isfinite(v)]
        if not cos_valid or not cka_valid:
            return HypothesisResult(
                hypothesis_id="H4",
                name="H4_task_specificity",
                description="Control: cos≥0.9998 AND CKA≥0.9998",
                passed=None,
                evidence="Insufficient control data.",
                metrics={},
            )
        mean_cos = float(np.mean(cos_valid))
        mean_cka = float(np.mean(cka_valid))
        passed = mean_cos >= 0.9998 and mean_cka >= 0.9998
        return HypothesisResult(
            hypothesis_id="H4",
            name="H4_task_specificity",
            description="Control: cos≥0.9998 AND CKA≥0.9998",
            passed=passed,
            evidence=f"control mean_cos={mean_cos:.5f}, mean_cka={mean_cka:.5f}",
            metrics={"mean_cosine": mean_cos, "mean_cka": mean_cka},
        )
