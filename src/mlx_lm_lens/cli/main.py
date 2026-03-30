"""Main CLI entry point with typer app and global flags."""

import logging
import sys

import typer

from mlx_lm_lens.cli.activations import activations_app
from mlx_lm_lens.cli.circuit import circuit_app
from mlx_lm_lens.cli.compare import compare
from mlx_lm_lens.cli.logit_lens import logit_lens_app

app = typer.Typer(help="Mechanistic interpretability CLI for MLX language models")
app.add_typer(logit_lens_app, name="logit-lens")
app.add_typer(activations_app, name="activations")
app.add_typer(circuit_app, name="circuit")
app.command(name="compare")(compare)


def setup_logging(verbose: bool, quiet: bool) -> None:
  """Configure logging level based on flags.

  Args:
    verbose: Enable verbose (DEBUG) logging.
    quiet: Suppress info logging.
  """
  if quiet:
    level = logging.WARNING
  elif verbose:
    level = logging.DEBUG
  else:
    level = logging.INFO

  logging.basicConfig(
    level=level,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    handlers=[logging.StreamHandler(sys.stderr)],
  )


@app.callback(invoke_without_command=True)
def main(
  ctx: typer.Context,
  verbose: bool = typer.Option(
    False,
    "--verbose",
    "-v",
    help="Enable verbose logging (DEBUG level)",
  ),
  quiet: bool = typer.Option(
    False,
    "--quiet",
    "-q",
    help="Suppress info logging (WARNING level only)",
  ),
) -> None:
  """mlx-lm-lens: Mechanistic interpretability tools for MLX language models.

  Provides layer-by-layer analysis of transformer models running on Apple
  Silicon via MLX, including logit lens, activation analysis, and circuit
  discovery.

  Use 'mlx-lm-lens <command> --help' for detailed command help.
  """
  setup_logging(verbose, quiet)

  if ctx.invoked_subcommand is None:
    typer.echo(ctx.get_help())


if __name__ == "__main__":
  app()
