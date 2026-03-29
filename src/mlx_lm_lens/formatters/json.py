"""JSON formatter for structured data output.

Formats analysis results as indented JSON with metadata (timestamp, metrics used).
Enables programmatic consumption of results by other tools.
"""

import json
from datetime import datetime
from typing import Any, Dict, List


class JsonFormatter:
    """Format analysis results as structured JSON."""

    def format_comparison_table(
        self,
        metrics: Dict[str, List[float]],
        layer_names: List[str],
    ) -> str:
        """Format metrics as JSON.

        Args:
            metrics: Dict mapping metric names to per-layer values.
            layer_names: List of layer identifiers.

        Returns:
            JSON string with metadata and metrics.
        """
        data = {
            "timestamp": datetime.now().isoformat(),
            "type": "comparison_table",
            "layer_names": layer_names,
            "metrics": metrics,
        }
        return json.dumps(data, indent=2)

    def format_logit_lens_results(self, results: Dict[str, Any]) -> str:
        """Format logit lens results as JSON.

        Args:
            results: Dict with layer outputs and predictions.

        Returns:
            JSON string with metadata and results.
        """
        data = {
            "timestamp": datetime.now().isoformat(),
            "type": "logit_lens_results",
            "results": results,
        }
        return json.dumps(data, indent=2, default=str)

    def format_activation_analysis(self, results: Dict[str, Any]) -> str:
        """Format activation analysis results as JSON.

        Args:
            results: Dict with metrics and layer information.

        Returns:
            JSON string with metadata and results.
        """
        data = {
            "timestamp": datetime.now().isoformat(),
            "type": "activation_analysis",
            "results": results,
        }
        return json.dumps(data, indent=2, default=str)

    def format_ablation_results(self, results: Dict[str, Any]) -> str:
        """Format ablation study results as JSON.

        Args:
            results: Dict with ablation scores and impacts.

        Returns:
            JSON string with metadata and results.
        """
        data = {
            "timestamp": datetime.now().isoformat(),
            "type": "ablation_results",
            "results": results,
        }
        return json.dumps(data, indent=2, default=str)
