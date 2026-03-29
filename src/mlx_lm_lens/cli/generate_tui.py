"""Rich TUI for streaming logit-lens generation."""

from collections import Counter

from rich.layout import Layout
from rich.panel import Panel
from rich.table import Table
from rich.text import Text

from mlx_lm_lens.core.logit_sampling import TokenResult


class GenerateTUI:
  """Live TUI display for token generation."""

  def __init__(self, model_name: str, prompt: str, max_tokens: int, sampling_method: str):
    self.model_name = model_name
    self.prompt = prompt
    self.max_tokens = max_tokens
    self.sampling_method = sampling_method
    self._results: list[TokenResult] = []
    self._layer_agreement_pct: float = 0.0
    self._max_visible_tokens = 200

    self.layout = Layout()
    self.layout.split_column(
      Layout(name="header", size=3),
      Layout(name="body"),
      Layout(name="layers", size=12),
    )
    self.layout["body"].split_row(
      Layout(name="stream", ratio=2),
      Layout(name="stats", ratio=1),
    )
    self._update_header()

  def _update_header(self) -> None:
    prompt_str = f"[bold cyan]{self.model_name}[/bold cyan] | {self.sampling_method}\nPrompt: {self.prompt[:50]}"
    self.layout["header"].update(Panel(prompt_str, title="Config"))

  def _update_stream(self) -> None:
    """Display generated text with scrolling."""
    if not self._results:
      self.layout["stream"].update(Panel("", title=f"Generated (0 tokens)"))
      return

    # Use the full decoded text from the last result (has all tokens with proper spacing)
    generated_text = self._results[-1].generated_text

    self.layout["stream"].update(Panel(generated_text, title=f"Generated ({len(self._results)} tokens)"))

  def _update_stats(self, result: TokenResult) -> None:
    n_tokens = len(self._results)
    elapsed = result.elapsed_sec
    speed = n_tokens / max(elapsed, 0.1)

    table = Table(show_header=False, box=None, padding=(0, 1))
    table.add_row("Tokens", f"{n_tokens} / {self.max_tokens}")
    table.add_row("Speed", f"{speed:.1f} tok/s")
    table.add_row("Elapsed", f"{elapsed:.1f}s")
    table.add_row("Agreement", f"{self._layer_agreement_pct:.0f}%")

    if result.drift_angles:
      mean_drift = sum(result.drift_angles.values()) / len(result.drift_angles)
      max_drift = max(result.drift_angles.values())
      corrected_count = sum(1 for a in result.drift_angles.values() if a > 0.3)

      table.add_row("Drift (mean)", f"{mean_drift:.2f}°")
      table.add_row("Drift (max)", f"{max_drift:.2f}°")
      table.add_row("Corrected", f"{corrected_count}/{len(result.drift_angles)}")

    self.layout["stats"].update(Panel(table, title="Stats"))

  def _update_layers(self, result: TokenResult) -> None:
    if not result.layer_predictions:
      self.layout["layers"].update(Panel("(waiting for logs)", title="Layers"))
      return

    has_drift = result.drift_angles is not None and len(result.drift_angles) > 0
    table = Table(title="Last Token Layer Predictions")
    table.add_column("Layer", style="cyan")
    table.add_column("#1", style="green")
    table.add_column("#2", style="yellow")
    table.add_column("#3", style="dim")
    if has_drift:
      table.add_column("Drift", style="white")

    layers = sorted(result.layer_predictions.keys())
    if len(layers) > 10:
      step = max(1, len(layers) // 10)
      layers = layers[::step]

    final_token = result.token_str
    agreement_count = 0
    total_layers = len(result.layer_predictions)

    for layer_key in layers:
      preds = result.layer_predictions[layer_key]
      top_1 = preds[0]["token"] if preds else "—"
      top_2 = preds[1]["token"] if len(preds) > 1 else "—"
      top_3 = preds[2]["token"] if len(preds) > 2 else "—"

      if top_1 == final_token:
        agreement_count += 1

      row = [layer_key, top_1, top_2, top_3]

      if has_drift:
        layer_idx = int(layer_key.split("_")[1])
        angle = result.drift_angles.get(layer_idx, 0.0)
        if angle < 0.3:
          angle_str = f"[green]{angle:.1f}°[/green]"
        elif angle < 1.0:
          angle_str = f"[yellow]{angle:.1f}°[/yellow]"
        else:
          angle_str = f"[red]{angle:.1f}°[/red]"
        row.append(angle_str)

      table.add_row(*row)

    self._layer_agreement_pct = 100.0 * agreement_count / max(total_layers, 1)
    self.layout["layers"].update(table)

  def update(self, result: TokenResult) -> None:
    """Update TUI with new token result."""
    self._results.append(result)
    self._update_stream()
    self._update_stats(result)
    self._update_layers(result)

  def get_layout(self) -> Layout:
    """Return the current layout."""
    return self.layout

  def get_generated_text(self) -> str:
    """Return all generated text so far."""
    if not self._results:
      return ""
    return self._results[-1].generated_text

  def build_token_frequency(self, all_results: list[TokenResult] | None = None) -> dict[str, float]:
    """Build frequency counter from all layer predictions."""
    if all_results is None:
      all_results = self._results

    freq = Counter()
    for result in all_results:
      for layer_preds in result.layer_predictions.values():
        for pred in layer_preds:
          token = pred["token"]
          prob = pred["probability"]
          freq[token] += prob

    return dict(freq)
