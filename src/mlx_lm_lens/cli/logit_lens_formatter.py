"""Output formatting for logit lens results."""

import csv
import io
import json
from pathlib import Path
from typing import Any, Dict

from rich.console import Console
from rich.table import Table

console = Console()


def format_output(result: Dict[str, Any], format_type: str) -> str:
  """Format logit lens results for output.

  Args:
    result: Results from LogitProjector.run_logit_lens().
    format_type: Format type (table, json, or csv).

  Returns:
    Formatted output string.
  """
  if format_type == "json":
    return json.dumps(_serialize_result(result), indent=2)

  if format_type == "csv":
    return _format_csv(result)

  return _format_table(result)


def _serialize_result(result: Dict[str, Any]) -> Dict[str, Any]:
  """Convert result to JSON-serializable format."""
  top_k = result["top_k_per_layer"]
  serialized_top_k = {}

  for layer, positions in top_k.items():
    serialized_top_k[str(layer)] = {}
    for pos, tokens in positions.items():
      serialized_top_k[str(layer)][str(pos)] = [
        {"token": t[0], "prob": t[1], "id": t[2]} for t in tokens
      ]

  return {
    "prompt": result["prompt"],
    "tokens": [{"text": t[0], "id": t[1]} for t in result["tokens"]],
    "top_k_per_layer": serialized_top_k,
    "flips": {
      str(pos): [
        {"layer_from": f[0], "layer_to": f[1], "old": f[2], "new": f[3]}
        for f in flips
      ]
      for pos, flips in result["flips"].items()
    },
  }


def _get_display_position(result: Dict[str, Any]) -> int:
  """Get position to display (last or first)."""
  return -1 if not result["config"]["all_positions"] else 0


def _format_table(result: Dict[str, Any]) -> str:
  """Format result as rich table."""
  table = Table(title=f"Logit Lens: {result['prompt'][:50]}")
  table.add_column("Layer", style="cyan")

  for col in range(result["config"]["top_k"]):
    table.add_column(f"Token {col + 1}", style="magenta")
    table.add_column("Prob", style="green")

  top_k = result["top_k_per_layer"]
  pos = _get_display_position(result)

  for layer in sorted(top_k.keys()):
    if pos not in top_k[layer]:
      continue

    tokens = top_k[layer][pos]
    row = [str(layer)]

    for token_str, prob, _ in tokens:
      row.append(token_str)
      row.append(f"{prob:.3f}")

    table.add_row(*row)

  if result["flips"]:
    table.add_section()
    for pos, flips in result["flips"].items():
      for layer_from, layer_to, old, new in flips:
        table.add_row(
          f"[red]FLIP[/red] L{layer_from}→L{layer_to}",
          f"{old}→{new}",
        )

  console.print(table)
  return ""


def _format_csv(result: Dict[str, Any]) -> str:
  """Format result as CSV with proper escaping."""
  output = io.StringIO()
  writer = csv.writer(output)
  writer.writerow(["layer", "position", "rank", "token", "probability", "token_id"])

  top_k = result["top_k_per_layer"]
  pos = _get_display_position(result)

  for layer in sorted(top_k.keys()):
    if pos not in top_k[layer]:
      continue

    tokens = top_k[layer][pos]
    for rank, (token_str, prob, token_id) in enumerate(tokens, 1):
      writer.writerow([layer, pos, rank, token_str, f"{prob:.6f}", token_id])

  return output.getvalue().rstrip("\r\n")


def write_output(content: str, output_file: str | None) -> None:
  """Write output to file or stdout.

  Args:
    content: Content to write.
    output_file: Optional file path. If None, prints via console.
  """
  if not content:
    return

  if output_file:
    Path(output_file).write_text(content)
    console.print(f"[green]Output written to {output_file}[/green]")
  else:
    console.print(content)
