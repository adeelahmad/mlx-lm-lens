"""Table formatter for terminal output using rich library.

Formats analysis results as rich-styled tables with colors for visual distinction
between good metrics (green) and anomalies (red).
"""

from io import StringIO
from typing import Any, Dict, List

from rich.console import Console
from rich.table import Table


class TableFormatter:
    """Format analysis results as rich-styled terminal tables."""

    def format_comparison_table(
        self,
        metrics: Dict[str, List[float]],
        layer_names: List[str],
    ) -> str:
        """Format metrics as a colored table.

        Args:
            metrics: Dict mapping metric names to lists of per-layer values.
            layer_names: List of layer identifiers.

        Returns:
            ANSI-formatted table string.
        """
        if not layer_names or not metrics:
            return ""

        table = Table(title="Metrics Comparison")
        table.add_column("Layer", style="cyan")

        for metric_name in metrics.keys():
            table.add_column(metric_name, justify="right")

        for i, layer_name in enumerate(layer_names):
            row = [layer_name]
            for metric_name in metrics.keys():
                if i < len(metrics[metric_name]):
                    value = metrics[metric_name][i]
                    formatted = f"{value:.4f}"
                    row.append(formatted)
                else:
                    row.append("N/A")

            table.add_row(*row)

        console = Console(file=StringIO(), force_terminal=True, width=100)
        console.print(table)
        return console.file.getvalue()

    def format_logit_lens_results(self, results: Dict[str, Any]) -> str:
        """Format logit lens results as a table.

        Args:
            results: Dict with layer outputs and top token predictions.

        Returns:
            Formatted table string.
        """
        if not results:
            return ""

        table = Table(title="Logit Lens Results")
        table.add_column("Layer", style="cyan")
        table.add_column("Top Token", style="green")
        table.add_column("Probability", justify="right")

        for layer_key, layer_data in sorted(results.items()):
            if isinstance(layer_data, dict) and "top_tokens" in layer_data:
                top_tokens = layer_data["top_tokens"]
                if top_tokens:
                    first_token = top_tokens[0]
                    token = first_token.get("token", "?")
                    prob = first_token.get("prob", 0.0)
                    table.add_row(layer_key, token, f"{prob:.4f}")

        console = Console(file=StringIO(), force_terminal=True, width=100)
        console.print(table)
        return console.file.getvalue()

    def format_activation_analysis(self, results: Dict[str, Any]) -> str:
        """Format activation analysis results.

        Args:
            results: Dict with metrics and layer information.

        Returns:
            Formatted table string.
        """
        if "metrics" not in results or "layer_names" not in results:
            return ""

        metrics = results["metrics"]
        layer_names = results["layer_names"]

        return self.format_comparison_table(metrics, layer_names)

    def format_ablation_results(self, results: Dict[str, Any]) -> str:
        """Format ablation study results.

        Args:
            results: Dict with ablation scores and layer impacts.

        Returns:
            Formatted table string.
        """
        if "ablated_layers" not in results or "impact_scores" not in results:
            return ""

        table = Table(title="Ablation Results")
        table.add_column("Layer", style="cyan")
        table.add_column("Impact Score", justify="right")

        ablated_layers = results["ablated_layers"]
        impact_scores = results["impact_scores"]

        for layer, score in zip(ablated_layers, impact_scores):
            formatted_score = f"{score:.4f}"
            table.add_row(str(layer), formatted_score)

        console = Console(file=StringIO(), force_terminal=True, width=100)
        console.print(table)
        return console.file.getvalue()
