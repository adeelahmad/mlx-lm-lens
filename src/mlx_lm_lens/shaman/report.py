"""SHAMAN hypothesis report generation."""

from typing import List
from mlx_lm_lens.shaman.validators import HypothesisResult


class SHAMANReporter:
    """Generate formatted SHAMAN hypothesis validation reports."""

    @staticmethod
    def generate_report(hypothesis_results: List[HypothesisResult]) -> str:
        """Generate text report from hypothesis validation results.

        Args:
            hypothesis_results: List of HypothesisResult objects from validators.

        Returns:
            Formatted text report with scoreboard, evidence, and summary stats.
        """
        if not hypothesis_results:
            return "No hypothesis results to report."

        lines: List[str] = []
        lines.append("=" * 80)
        lines.append("SHAMAN HYPOTHESIS VALIDATION REPORT")
        lines.append("=" * 80)
        lines.append("")

        # Scoreboard
        lines.append("HYPOTHESIS SCOREBOARD")
        lines.append("-" * 80)
        passed_count = 0
        failed_count = 0
        na_count = 0
        for result in hypothesis_results:
            if result.passed is True:
                badge = "PASS"
                passed_count += 1
            elif result.passed is False:
                badge = "FAIL"
                failed_count += 1
            else:
                badge = "N/A"
                na_count += 1
            lines.append(f"  [{badge:^4}] {result.hypothesis_id:>3} | {result.name}")
        lines.append("")

        # Per-hypothesis evidence
        lines.append("DETAILED EVIDENCE")
        lines.append("-" * 80)
        for result in hypothesis_results:
            lines.append(f"\n{result.hypothesis_id} - {result.name}")
            lines.append(f"  Description: {result.description}")
            lines.append(f"  Evidence: {result.evidence}")
            if result.metrics:
                for key, val in result.metrics.items():
                    lines.append(f"  • {key}: {val:.6g}")

        # Summary
        lines.append("")
        lines.append("=" * 80)
        lines.append("SUMMARY")
        lines.append("-" * 80)
        total = len(hypothesis_results)
        lines.append(f"  Passed: {passed_count}/{total}")
        lines.append(f"  Failed: {failed_count}/{total}")
        lines.append(f"  N/A:    {na_count}/{total}")
        pass_rate = (passed_count / total * 100) if total > 0 else 0.0
        lines.append(f"  Pass Rate: {pass_rate:.1f}%")
        lines.append("=" * 80)

        return "\n".join(lines)
