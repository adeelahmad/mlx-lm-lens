"""CSV formatter for tabular data export.

Formats analysis results as CSV with proper headers and escaping.
Suitable for import into spreadsheet applications.
"""

import csv
from io import StringIO
from typing import Any, Dict, List


class CsvFormatter:
    """Format analysis results as CSV."""

    def format_comparison_table(
        self,
        metrics: Dict[str, List[float]],
        layer_names: List[str],
    ) -> str:
        """Format metrics as CSV.

        Args:
            metrics: Dict mapping metric names to per-layer values.
            layer_names: List of layer identifiers.

        Returns:
            CSV string with headers and data rows.
        """
        if not layer_names or not metrics:
            return ""

        output = StringIO()
        metric_names = list(metrics.keys())
        fieldnames = ["layer"] + metric_names

        writer = csv.DictWriter(output, fieldnames=fieldnames)
        writer.writeheader()

        for i, layer_name in enumerate(layer_names):
            row = {"layer": layer_name}
            for metric_name in metric_names:
                if i < len(metrics[metric_name]):
                    row[metric_name] = metrics[metric_name][i]
                else:
                    row[metric_name] = ""
            writer.writerow(row)

        return output.getvalue()

    def format_logit_lens_results(self, results: Dict[str, Any]) -> str:
        """Format logit lens results as CSV.

        Args:
            results: Dict with layer outputs and predictions.

        Returns:
            CSV string with headers and data rows.
        """
        if not results:
            return ""

        output = StringIO()
        fieldnames = ["layer", "top_token", "probability"]
        writer = csv.DictWriter(output, fieldnames=fieldnames)
        writer.writeheader()

        for layer_key, layer_data in sorted(results.items()):
            if isinstance(layer_data, dict) and "top_tokens" in layer_data:
                top_tokens = layer_data["top_tokens"]
                if top_tokens:
                    first_token = top_tokens[0]
                    token = first_token.get("token", "")
                    prob = first_token.get("prob", "")
                    writer.writerow(
                        {
                            "layer": layer_key,
                            "top_token": token,
                            "probability": prob,
                        }
                    )

        return output.getvalue()

    def format_activation_analysis(self, results: Dict[str, Any]) -> str:
        """Format activation analysis results as CSV.

        Args:
            results: Dict with metrics and layer information.

        Returns:
            CSV string with headers and data rows.
        """
        if "metrics" not in results or "layer_names" not in results:
            return ""

        metrics = results["metrics"]
        layer_names = results["layer_names"]

        return self.format_comparison_table(metrics, layer_names)

    def format_ablation_results(self, results: Dict[str, Any]) -> str:
        """Format ablation study results as CSV.

        Args:
            results: Dict with ablation scores and impacts.

        Returns:
            CSV string with headers and data rows.
        """
        if "ablated_layers" not in results or "impact_scores" not in results:
            return ""

        output = StringIO()
        fieldnames = ["layer", "impact_score"]
        writer = csv.DictWriter(output, fieldnames=fieldnames)
        writer.writeheader()

        ablated_layers = results["ablated_layers"]
        impact_scores = results["impact_scores"]

        for layer, score in zip(ablated_layers, impact_scores):
            writer.writerow({"layer": layer, "impact_score": score})

        return output.getvalue()
