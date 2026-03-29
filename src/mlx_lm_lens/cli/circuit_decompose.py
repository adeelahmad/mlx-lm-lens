"""Circuit decompose command: residual stream decomposition."""

import logging

import typer
from rich.console import Console
from rich.table import Table

from mlx_lm_lens.circuit.decompose import ResidualStreamDecomposer
from mlx_lm_lens.core.model_loader import ModelLoader
from mlx_lm_lens.core.tokenizer_utils import load_tokenizer
from mlx_lm_lens.exceptions import ConfigError, ModelLoadError

logger = logging.getLogger(__name__)
console = Console()


def decompose(
  model: str = typer.Option(..., "--model", help="Path to MLX model"),
  prompt: str = typer.Option(..., "--prompt", help="Input prompt"),
  position: int = typer.Option(-1, "--position", help="Token position (-1=last)"),
  show_contributions: bool = typer.Option(
    False,
    "--show-contributions",
    help="Show per-layer contributions",
  ),
) -> None:
  """Residual stream decomposition.

  Show how each layer contributes additively to the residual stream.

  Example:
    mlx-lm-lens circuit decompose --model /path/to/model --prompt "..."
  """
  try:
    logger.info(f"Loading model: {model}")
    loaded_model, _ = ModelLoader.load(model)

    logger.info("Loading tokenizer")
    tokenizer = load_tokenizer(model)

    logger.info("Computing residual stream decomposition")
    result = ResidualStreamDecomposer.show_contributions(
      loaded_model,
      tokenizer,
      prompt,
      position,
    )

    # Display results
    table = Table(title=f"Residual Contributions (position={position})")
    table.add_column("Layer", style="magenta")
    table.add_column("L2 Norm", style="green")
    table.add_column("Normalized", style="cyan")

    raw = result["raw_norms"]
    normalized = result["normalized_scores"]

    for layer_idx in sorted(raw.keys()):
      table.add_row(
        str(layer_idx),
        f"{raw[layer_idx]:.4f}",
        f"{normalized[layer_idx]:.4f}",
      )

    console.print(table)

  except (ConfigError, ModelLoadError) as e:
    logger.error(f"Error: {e}")
    typer.Exit(code=1)
