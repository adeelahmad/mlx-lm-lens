"""Formatter protocol defining the output formatting interface.

All concrete formatters (table, JSON, CSV) must implement this Protocol.
This enables swappable output backends without coupling analysis code to
specific output formats.
"""

from typing import Any, Dict, List, Protocol


class Formatter(Protocol):
    """Output formatter interface for analysis results.

    Defines methods for formatting different types of analysis results
    (comparison tables, logit lens, activation analysis, ablation results)
    into various output formats (table, JSON, CSV).
    """

    def format_comparison_table(
        self,
        metrics: Dict[str, List[float]],
        layer_names: List[str],
    ) -> str:
        """Format metrics comparison table.

        Args:
            metrics: Dict mapping metric names to lists of values per layer.
                    E.g., {"cosine": [0.95, 0.92, ...], "cka": [...]}
            layer_names: List of layer identifiers (e.g., ["layer_0", ...])

        Returns:
            Formatted string representation (table, JSON, or CSV).
        """
        ...

    def format_logit_lens_results(self, results: Dict[str, Any]) -> str:
        """Format logit lens analysis results.

        Args:
            results: Dict containing logit lens output with structure like:
                    {
                        "layer_N": {
                            "token_position": int,
                            "top_tokens": [{"token": str, "prob": float}, ...]
                        },
                        ...
                    }

        Returns:
            Formatted string representation.
        """
        ...

    def format_activation_analysis(self, results: Dict[str, Any]) -> str:
        """Format activation analysis results.

        Args:
            results: Dict containing activation metrics with structure like:
                    {
                        "metrics": {"cosine": [...], "cka": [...], ...},
                        "layer_names": [...],
                        "model_pair": (model_a, model_b)
                    }

        Returns:
            Formatted string representation.
        """
        ...

    def format_ablation_results(self, results: Dict[str, Any]) -> str:
        """Format ablation study results.

        Args:
            results: Dict containing ablation results with structure like:
                    {
                        "ablated_layers": [0, 1, 2, ...],
                        "impact_scores": [...],
                        "baseline_loss": float,
                        "ablation_loss": float
                    }

        Returns:
            Formatted string representation.
        """
        ...
