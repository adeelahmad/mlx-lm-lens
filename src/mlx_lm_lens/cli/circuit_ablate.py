"""Circuit ablate command: measure layer importance via ablation."""

import json
import logging
from pathlib import Path
from typing import Optional

import typer
from rich.console import Console
from rich.table import Table

from mlx_lm_lens.circuit.ablation_sweep import AblationSweep
from mlx_lm_lens.core.model_loader import ModelLoader
from mlx_lm_lens.core.tokenizer_utils import load_tokenizer
from mlx_lm_lens.exceptions import ConfigError, ModelLoadError

logger = logging.getLogger(__name__)
console = Console()


def ablate(
  model: str = typer.Option(..., "--model", help="Path to MLX model"),
  prompt: str = typer.Option(..., "--prompt", help="Input prompt"),
  method: str = typer.Option(
    "zero",
    "--method",
    help="Ablation method: zero, mean, noise, knockout",
  ),
  target_token: Optional[str] = typer.Option(
    None,
    "--target-token",
    help="Token to measure probability for",
  ),
  layers: Optional[str] = typer.Option(
    None,
    "--layers",
    help="Layer range: 20-35 (start-end inclusive)",
  ),
  format_type: str = typer.Option("table", "--format", help="Output format"),
  output: Optional[str] = typer.Option(None, "--output", help="Output file path"),
) -> None:
  """Ablate layers and measure importance via KL divergence.

  Example:
    mlx-lm-lens circuit ablate --model /path/to/model --prompt "Who are you?"
  """
  try:
    if method not in ("zero", "mean", "noise", "knockout"):
      raise ConfigError(f"Invalid method: {method}")

    logger.info(f"Loading model: {model}")
    loaded_model, config = ModelLoader.load(model)

    logger.info("Loading tokenizer")
    tokenizer = load_tokenizer(model)

    # Parse layer range
    layer_range = None
    if layers:
      try:
        start, end = map(int, layers.split("-"))
        layer_range = (start, end)
      except ValueError:
        raise ConfigError(f"Invalid layer range format: {layers}")

    logger.info(f"Running ablation sweep with method={method}")
    importance = AblationSweep.sweep_all_layers(
      loaded_model,
      tokenizer,
      prompt,
      method=method,
      target_token=target_token,
      layer_range=layer_range,
    )

    ranked = AblationSweep.rank_layers(importance)

    # Format output
    if format_type == "json":
      output_dict = {
        "method": method,
        "layers": {str(idx): score for idx, score in importance.items()},
        "ranked": [
          {"layer": idx, "importance": score, "rank": rank}
          for idx, score, rank in ranked
        ],
      }
      output_str = json.dumps(output_dict, indent=2)
    else:  # table format
      table = Table(title="Layer Importance (Ablation)")
      table.add_column("Rank", style="cyan")
      table.add_column("Layer", style="magenta")
      table.add_column("Importance", style="green")

      for idx, score, rank in ranked:
        table.add_row(str(rank), str(idx), f"{score:.4f}")

      console.print(table)
      output_str = None

    if output_str:
      if output:
        Path(output).write_text(output_str)
        logger.info(f"Output written to {output}")
      else:
        console.print(output_str)

  except (ConfigError, ModelLoadError) as e:
    logger.error(f"Error: {e}")
    typer.Exit(code=1)
