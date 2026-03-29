"""Format and output model comparison results."""

import json
import logging
from pathlib import Path
from typing import Any

from rich.console import Console
from rich.table import Table

logger = logging.getLogger(__name__)
console = Console()


def serialize_single_result(result: dict[str, Any]) -> dict[str, Any]:
  """Convert run_logit_lens result to JSON-serializable form.

  Args:
    result: Raw result from LogitProjector.run_logit_lens().

  Returns:
    JSON-serializable dict with stringified keys.
  """
  top_k_per_layer: dict[str, dict[str, list[dict[str, Any]]]] = {}

  for layer_idx, positions in result.get("top_k_per_layer", {}).items():
    layer_key = str(layer_idx)
    top_k_per_layer[layer_key] = {}

    for pos, tokens in positions.items():
      pos_key = str(pos)
      top_k_per_layer[layer_key][pos_key] = [
        {"token": tok[0], "prob": float(tok[1]), "id": int(tok[2])} for tok in tokens
      ]

  flips: dict[str, list[dict[str, Any]]] = {}
  for pos, flip_list in result.get("flips", {}).items():
    pos_key = str(pos)
    flips[pos_key] = [
      {"layer_from": f[0], "layer_to": f[1], "old": f[2], "new": f[3]} for f in flip_list
    ]

  return {
    "prompt": result.get("prompt", ""),
    "tokens": [{"text": t[0], "id": t[1]} for t in result.get("tokens", [])],
    "top_k_per_layer": top_k_per_layer,
    "flips": flips,
  }


def _derive_output_paths(output_path: str) -> tuple[str, str, str]:
  """Derive _base, _adapted, _diff paths from base output path.

  Args:
    output_path: User-provided --output path (e.g., "result.json").

  Returns:
    Tuple of (base_path, adapted_path, diff_path).
  """
  path = Path(output_path)
  stem = path.stem
  suffix = path.suffix
  parent = path.parent

  base_path = str(parent / f"{stem}_base{suffix}")
  adapted_path = str(parent / f"{stem}_adapted{suffix}")
  diff_path = str(parent / f"{stem}_diff{suffix}")

  return base_path, adapted_path, diff_path


def write_comparison_files(
    base_result: dict[str, Any],
    adapted_result: dict[str, Any],
    diff: dict[str, Any],
    output_path: str,
) -> tuple[str, str, str]:
  """Write base, adapted, and diff JSON files.

  Args:
    base_result: Serialized base model result.
    adapted_result: Serialized adapted model result.
    diff: Diff dict from compute_diff().
    output_path: User-provided --output path.

  Returns:
    Tuple of (base_path, adapted_path, diff_path) written.
  """
  base_path, adapted_path, diff_path = _derive_output_paths(output_path)

  Path(base_path).write_text(json.dumps(serialize_single_result(base_result), indent=2))
  Path(adapted_path).write_text(json.dumps(serialize_single_result(adapted_result), indent=2))
  Path(diff_path).write_text(json.dumps(diff, indent=2))

  console.print("[green]✓ Results saved:[/green]")
  console.print(f"  Base:    {base_path}")
  console.print(f"  Adapted: {adapted_path}")
  console.print(f"  Diff:    {diff_path}")

  return base_path, adapted_path, diff_path


def format_comparison_table(
    diff: dict[str, Any],
    base_result: dict[str, Any],
    adapted_result: dict[str, Any],
) -> None:
  """Print side-by-side comparison table to console.

  Args:
    diff: Output of compute_diff().
    base_result: Base model result dict.
    adapted_result: Adapted model result dict.
  """
  table = Table(title="Model Comparison: Base vs Adapted")
  table.add_column("Layer", style="cyan")
  table.add_column("Pos", style="cyan")
  table.add_column("Base Top-1", style="white")
  table.add_column("Base Prob", style="dim")
  table.add_column("Adapted Top-1", style="white")
  table.add_column("Adapted Prob", style="dim")
  table.add_column("Changed", style="white")

  changed_count = 0
  total_positions = 0

  for layer_key in sorted(diff.get("layers", {}).keys(), key=lambda x: int(x)):
    positions = diff["layers"][layer_key]

    for pos_key in sorted(positions.keys(), key=lambda x: int(x)):
      pos_info = positions[pos_key]
      total_positions += 1

      base_top1 = pos_info["base_top1"]["token"]
      base_prob = pos_info["base_top1"]["prob"]
      adapted_top1 = pos_info["adapted_top1"]["token"]
      adapted_prob = pos_info["adapted_top1"]["prob"]
      changed = pos_info["top1_changed"]

      if changed:
        changed_count += 1
        style = "red"
        changed_str = "✗"
      else:
        style = "green"
        changed_str = "✓"

      table.add_row(
        layer_key,
        pos_key,
        f"[{style}]{base_top1}[/{style}]",
        f"{base_prob:.3f}",
        f"[{style}]{adapted_top1}[/{style}]",
        f"{adapted_prob:.3f}",
        changed_str,
      )

  console.print(table)
  console.print(
    f"[yellow]Summary:[/yellow] {changed_count}/{total_positions} positions "
    f"changed top-1 prediction"
  )
