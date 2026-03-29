"""Circuit angles command: weight angle analysis between checkpoints."""

import json
import logging

import typer
from rich.console import Console
from rich.table import Table

from mlx_lm_lens.circuit.angles import WeightAngleAnalyzer

logger = logging.getLogger(__name__)
console = Console()


def angles(
  base_adapter: str = typer.Option(..., "--base-adapter", help="Base checkpoint"),
  current_adapter: str = typer.Option(
    ...,
    "--current-adapter",
    help="Current checkpoint",
  ),
  per_layer: bool = typer.Option(
    False,
    "--per-layer",
    help="Show per-layer angles",
  ),
  format_type: str = typer.Option("table", "--format", help="Output format"),
) -> None:
  """Weight angle analysis between checkpoints.

  Measure how much model weights rotated during training.

  Example:
    mlx-lm-lens circuit angles --base-adapter /path/a --current-adapter /path/b
  """
  try:
    logger.info("Computing weight angles")
    angles_dict = WeightAngleAnalyzer.compute_weight_angles(
      base_adapter,
      current_adapter,
      per_layer=per_layer,
    )

    if format_type == "json":
      output_str = json.dumps(
        {k: float(v) for k, v in angles_dict.items()},
        indent=2,
      )
      console.print(output_str)
    else:  # table
      if per_layer:
        table = Table(title="Weight Angles (Per-Layer)")
        table.add_column("Weight Name", style="magenta")
        table.add_column("Angle (degrees)", style="green")

        for weight_name in sorted(angles_dict.keys()):
          table.add_row(
            weight_name,
            f"{angles_dict[weight_name]:.2f}",
          )
      else:
        table = Table(title="Weight Angles (Aggregated)")
        table.add_row(
          "Mean Angle (degrees)",
          f"{angles_dict.get('aggregate', 0):.2f}",
        )

      console.print(table)

  except Exception as e:
    logger.error(f"Error: {e}")
    typer.Exit(code=1)
