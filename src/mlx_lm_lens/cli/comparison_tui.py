"""Rich TUI for displaying base vs adapted logit-lens comparison."""

from typing import Any

from rich.layout import Layout
from rich.panel import Panel
from rich.table import Table


class ComparisonTUI:
  """Rich TUI display for base vs adapted logit-lens comparison."""

  def __init__(
      self,
      model_name: str,
      adapter_path: str,
      prompt: str,
  ) -> None:
    """Initialize layout with header, body panels, and stats.

    Args:
      model_name: Base model identifier for header display.
      adapter_path: Adapter path for header display.
      prompt: Input prompt (truncated to 50 chars in header).
    """
    self.model_name = model_name
    self.adapter_path = adapter_path
    self.prompt = prompt
    self._changed_count = 0
    self._total_positions = 0

    self.layout = Layout()
    self.layout.split_column(
      Layout(name="header", size=3),
      Layout(name="body"),
      Layout(name="flip_diffs", size=8),
    )
    self.layout["body"].split_row(
      Layout(name="base_preds", ratio=2),
      Layout(name="adapted_preds", ratio=2),
      Layout(name="stats", ratio=1),
    )
    self._update_header()

  def _update_header(self) -> None:
    """Update header with model, adapter, and prompt info."""
    header_text = (
      f"[bold cyan]{self.model_name}[/bold cyan] | "
      f"[bold magenta]{self.adapter_path[:30]}[/bold magenta]\n"
      f"Prompt: {self.prompt[:50]}"
    )
    self.layout["header"].update(Panel(header_text, title="Config"))

  def _build_pred_table(
      self,
      result: dict[str, Any],
      diff: dict[str, Any],
      label: str,
      highlight_changed: bool,
  ) -> Table:
    """Build a Rich Table for one side (base or adapted) of the comparison.

    Args:
      result: Logit lens result dict.
      diff: Diff dict from compute_diff().
      label: "Base" or "Adapted" for styling.
      highlight_changed: If True, mark changed rows in red.

    Returns:
      Rich Table with layer and top-1 predictions.
    """
    table = Table(title=label, show_header=True, box=None)
    table.add_column("Layer", style="cyan")
    table.add_column("Top-1", style="white")

    is_base = label == "Base"

    for layer_key in sorted(diff.get("layers", {}).keys(), key=lambda x: int(x)):
      positions = diff["layers"][layer_key]

      for pos_key in sorted(positions.keys(), key=lambda x: int(x)):
        pos_info = positions[pos_key]

        if is_base:
          top1 = pos_info["base_top1"]["token"]
          changed = pos_info["top1_changed"]
        else:
          top1 = pos_info["adapted_top1"]["token"]
          changed = pos_info["top1_changed"]

        if highlight_changed and changed:
          top1_styled = f"[red]{top1}[/red]"
        elif not changed:
          top1_styled = f"[green]{top1}[/green]"
        else:
          top1_styled = top1

        table.add_row(layer_key, top1_styled)

    return table

  def _build_stats_table(self, diff: dict[str, Any]) -> Table:
    """Build a summary stats panel.

    Args:
      diff: Diff dict from compute_diff().

    Returns:
      Rich Table with summary statistics.
    """
    changed_count = 0
    total_positions = 0

    for layer_info in diff.get("layers", {}).values():
      for pos_info in layer_info.values():
        total_positions += 1
        if pos_info["top1_changed"]:
          changed_count += 1

    table = Table(show_header=False, box=None, padding=(0, 1))
    table.add_row("Positions", f"{total_positions}")
    table.add_row("Changed", f"{changed_count}/{total_positions}")
    table.add_row("Change %", f"{100*changed_count/max(total_positions, 1):.0f}%")

    return Panel(table, title="Stats")

  def _build_flip_diff_table(self) -> Table:
    """Build table showing positions where flip points differ.

    Returns:
      Rich Table with flip point diffs.
    """
    table = Table(title="Flip Point Diffs", show_header=True, box=None)
    table.add_column("Position", style="cyan")
    table.add_column("Info", style="white")

    table.add_row("—", "(Not shown in static TUI)")

    return table

  def update(
      self,
      diff: dict[str, Any],
      base_result: dict[str, Any],
      adapted_result: dict[str, Any],
  ) -> None:
    """Populate all panels from comparison results.

    Args:
      diff: Output of compute_diff().
      base_result: Base model run_logit_lens result.
      adapted_result: Adapted model run_logit_lens result.
    """
    self.layout["base_preds"].update(
      self._build_pred_table(base_result, diff, "Base", highlight_changed=False)
    )
    self.layout["adapted_preds"].update(
      self._build_pred_table(adapted_result, diff, "Adapted", highlight_changed=True)
    )
    self.layout["stats"].update(self._build_stats_table(diff))
    self.layout["flip_diffs"].update(self._build_flip_diff_table())

  def get_layout(self) -> Layout:
    """Return the fully populated layout for rendering.

    Returns:
      Rich Layout object ready to pass to Live() or Console.print().
    """
    return self.layout
