"""Word cloud rendering for logit-lens generation."""


from rich.console import Console
from rich.panel import Panel
from rich.text import Text


def render_wordcloud_text(freq: dict[str, float], top_n: int = 40) -> Panel:
  """Render a text-based word cloud using Rich formatting.

  Larger tokens get bold/magenta, smaller get dimmed.
  """
  if not freq:
    return Panel("(no tokens)", title="Word Cloud")

  items = sorted(freq.items(), key=lambda x: x[1], reverse=True)[:top_n]
  if not items:
    return Panel("(no tokens)", title="Word Cloud")

  max_freq = items[0][1]
  min_freq = items[-1][1]
  freq_range = max(max_freq - min_freq, 0.001)

  text = Text()
  for i, (token, freq_val) in enumerate(items):
    norm = (freq_val - min_freq) / freq_range
    if norm > 0.7:
      style = "bold magenta"
    elif norm > 0.4:
      style = "cyan"
    else:
      style = "dim"

    text.append(token, style=style)
    if (i + 1) % 8 == 0:
      text.append("\n")
    else:
      text.append("  ")

  return Panel(text, title="Word Cloud")


def render_wordcloud_matplotlib(freq: dict[str, float], output_path: str | None = None) -> None:
  """Render word cloud using matplotlib wordcloud package.

  Args:
    freq: Token frequency dict.
    output_path: Path to save PNG (if provided).

  Raises:
    ImportError: If wordcloud package not installed.
  """
  try:
    from wordcloud import WordCloud
  except ImportError as e:
    raise ImportError(
      "wordcloud package not installed. Install with: pip install wordcloud"
    ) from e

  if not freq:
    raise ValueError("No tokens to visualize")

  wc = WordCloud(width=1200, height=600, background_color="white")
  wc.generate_from_frequencies(freq)

  if output_path:
    wc.to_file(output_path)
  else:
    try:
      import matplotlib.pyplot as plt

      plt.figure(figsize=(12, 6))
      plt.imshow(wc, interpolation="bilinear")
      plt.axis("off")
      plt.tight_layout(pad=0)
      plt.show()
    except ImportError as e:
      raise ImportError("matplotlib not available for display") from e


def render_wordcloud(
  freq: dict[str, float],
  output_path: str | None,
  console: Console,
) -> None:
  """Render word cloud: try matplotlib → text fallback.

  Args:
    freq: Token frequency dict.
    output_path: Path to save PNG (if provided).
    console: Rich console for output.
  """
  if output_path:
    try:
      render_wordcloud_matplotlib(freq, output_path)
      console.print(f"[green]✓ Word cloud saved to {output_path}[/green]")
    except ImportError as e:
      console.print(f"[yellow]Warning: {e}[/yellow]")
      console.print(render_wordcloud_text(freq))
  else:
    try:
      render_wordcloud_matplotlib(freq)
    except ImportError:
      console.print(render_wordcloud_text(freq))
