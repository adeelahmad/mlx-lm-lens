"""Circuit discovery CLI subcommands: ablate, patch, decompose, angles."""

import logging

import typer

from mlx_lm_lens.cli.circuit_ablate import ablate
from mlx_lm_lens.cli.circuit_patch import patch
from mlx_lm_lens.cli.circuit_decompose import decompose
from mlx_lm_lens.cli.circuit_angles import angles

logger = logging.getLogger(__name__)

circuit_app = typer.Typer(help="Circuit discovery: ablation, patching, decomposition")
circuit_app.command()(ablate)
circuit_app.command()(patch)
circuit_app.command()(decompose)
circuit_app.command()(angles)
