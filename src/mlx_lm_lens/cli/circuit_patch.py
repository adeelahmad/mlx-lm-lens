"""Circuit patch command: activation patching between models."""

import json
import logging
from pathlib import Path
from typing import Optional

import typer
from rich.console import Console
from rich.table import Table

from mlx_lm_lens.circuit.patch import ActivationPatcher
from mlx_lm_lens.core.model_loader import ModelLoader
from mlx_lm_lens.core.tokenizer_utils import load_tokenizer
from mlx_lm_lens.exceptions import ConfigError, ModelLoadError

logger = logging.getLogger(__name__)
console = Console()


def patch(
  model: str = typer.Option(..., "--model", help="Primary model"),
  reference: str = typer.Option(..., "--reference", help="Reference model"),
  prompt: str = typer.Option(..., "--prompt", help="Input prompt"),
  patch_layer: Optional[int] = typer.Option(
    None,
    "--patch-layer",
    help="Single layer to patch",
  ),
  sweep: bool = typer.Option(False, "--sweep", help="Patch all layers"),
  format_type: str = typer.Option("table", "--format", help="Output format"),
  output: Optional[str] = typer.Option(None, "--output", help="Output file path"),
) -> None:
  """Activation patching between models.

  Replace layers from one model with another, measure KL divergence.

  Example:
    mlx-lm-lens circuit patch --model /path/to/model_a \\
      --reference /path/to/model_b --prompt "..."
  """
  try:
    if not sweep and patch_layer is None:
      raise ConfigError("Specify --patch-layer or use --sweep")

    logger.info(f"Loading primary model: {model}")
    model_a, _ = ModelLoader.load(model)

    logger.info(f"Loading reference model: {reference}")
    model_b, _ = ModelLoader.load(reference)

    logger.info("Loading tokenizer")
    tokenizer = load_tokenizer(model)

    if sweep:
      logger.info("Running patch sweep on all layers")
      kl_divs = ActivationPatcher.sweep_patch_all_layers(
        model_a,
        model_b,
        tokenizer,
        prompt,
      )

      if format_type == "json":
        output_str = json.dumps({"kl_divs": kl_divs}, indent=2)
      else:  # table
        table = Table(title="Layer Patching KL Divergence")
        table.add_column("Layer", style="magenta")
        table.add_column("KL Divergence", style="green")

        for layer_idx in sorted(kl_divs.keys()):
          table.add_row(str(layer_idx), f"{kl_divs[layer_idx]:.4f}")

        console.print(table)
        output_str = None

    else:
      logger.info(f"Patching layer {patch_layer}")
      result = ActivationPatcher.patch_layer(
        model_a,
        model_b,
        tokenizer,
        prompt,
        patch_layer,
      )

      if format_type == "json":
        output_str = json.dumps(
          {
            "layer": patch_layer,
            "kl": result.get("kl", 0.0),
            "token_flip": result.get("token_flip", False),
          },
          indent=2,
        )
      else:  # table
        table = Table(title=f"Layer {patch_layer} Patching Result")
        table.add_row("KL Divergence", f"{result.get('kl', 0.0):.4f}")
        table.add_row("Token Flip", str(result.get("token_flip", False)))
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
