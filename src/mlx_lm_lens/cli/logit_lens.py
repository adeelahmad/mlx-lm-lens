"""Logit lens CLI command."""

import logging
from typing import Any

import typer
from rich.console import Console

from mlx_lm_lens.cli.comparison_formatter import format_comparison_table, write_comparison_files
from mlx_lm_lens.cli.comparison_tui import ComparisonTUI
from mlx_lm_lens.cli.generate import generate
from mlx_lm_lens.cli.logit_lens_formatter import format_output, write_output
from mlx_lm_lens.config import LogitLensConfig
from mlx_lm_lens.core.comparison_engine import compute_diff
from mlx_lm_lens.core.logit_projection import LogitProjector
from mlx_lm_lens.core.model_loader import ModelLoader
from mlx_lm_lens.exceptions import ConfigError, ModelLoadError

logger = logging.getLogger(__name__)
console = Console()

logit_lens_app = typer.Typer(help="Analyze layer-wise logit outputs")


def _prepare_prompt(prompt: str, tokenizer: Any, use_chat_template: bool) -> str:
  """Apply chat template if requested, fallback to raw prompt on error."""
  if not use_chat_template or not hasattr(tokenizer, "apply_chat_template"):
    return prompt
  try:
    return tokenizer.apply_chat_template(
      [{"role": "user", "content": prompt}],
      tokenize=False,
    )
  except (AttributeError, TypeError, RuntimeError) as e:
    logger.warning(f"Chat template failed, using raw prompt: {e}")
    return prompt


def _run_single(
    model_path: str,
    adapter_path: str | None,
    prompt: str,
    config: LogitLensConfig,
) -> dict[str, Any]:
  """Load model, run logit lens, unload."""
  loaded_model = None
  try:
    logger.info(f"Loading model: {model_path}" + (f" + {adapter_path}" if adapter_path else ""))
    loaded_model, tokenizer = ModelLoader.load(model_path, adapter_path=adapter_path)
    result = LogitProjector.run_logit_lens(loaded_model, tokenizer, prompt, config)
    return result
  finally:
    if loaded_model is not None:
      try:
        ModelLoader.unload(loaded_model)
      except Exception as e:
        logger.warning(f"Error during cleanup: {e}")


def _run_comparison(prompt: str, config: LogitLensConfig) -> None:
  """Runs comparison: base, adapted, diff, files, table, TUI."""
  logger.info("Running base model...")
  base_result = _run_single(config.model_path, None, prompt, config)
  logger.info("Running adapted model...")
  adapted_result = _run_single(config.model_path, config.adapter_path, prompt, config)
  diff = compute_diff(base_result, adapted_result, config.model_path, config.adapter_path)
  if config.output_file:
    write_comparison_files(base_result, adapted_result, diff, config.output_file)
  format_comparison_table(diff, base_result, adapted_result)
  if config.output_file:
    tui = ComparisonTUI(config.model_path, config.adapter_path, prompt)
    tui.update(diff, base_result, adapted_result)
    console.print(tui.get_layout())


@logit_lens_app.command()
def logit_lens(
  model: str = typer.Option(
    ...,
    "--model",
    "-m",
    help="Path to MLX model or HuggingFace model ID",
  ),
  prompt: str = typer.Option(
    ...,
    "--prompt",
    "-p",
    help="Input text to analyze",
  ),
  adapter: str | None = typer.Option(
    None,
    "--adapter",
    "-a",
    help="Optional path to LoRA adapter",
  ),
  top_k: int = typer.Option(
    5,
    "--top-k",
    "-k",
    help="Number of top tokens to show per layer",
    min=1,
  ),
  all_positions: bool = typer.Option(
    False,
    "--all-positions",
    help="Analyze all token positions (default: last only)",
  ),
  compare_base: bool = typer.Option(
    False,
    "--compare-base",
    help="Compare base vs adapted (requires --adapter)",
  ),
  chat_template: bool = typer.Option(
    False,
    "--chat-template",
    help="Apply model's chat template to prompt",
  ),
  format_type: str = typer.Option(
    "table",
    "--format",
    "-f",
    help="Output format: table, json, or csv",
  ),
  output: str | None = typer.Option(
    None,
    "--output",
    "-o",
    help="Output file path (default: stdout)",
  ),
) -> None:
  """Analyze what each layer's hidden state 'wants to output' at each position.

  Shows top-K token probabilities per layer, revealing how predictions evolve
  through the network and where identity flips occur.

  Example:
    mlx-lm-lens logit-lens --model qwen3 --prompt "Who are you?"
  """
  loaded_model = None
  try:
    config = LogitLensConfig(
      model_path=model,
      prompt=prompt,
      adapter_path=adapter,
      top_k=top_k,
      all_positions=all_positions,
      compare_base=compare_base,
      use_chat_template=chat_template,
      format_type=format_type,
      output_file=output,
    )
    config.validate()

    tokenizer = None
    if not compare_base:
      loaded_model, tokenizer = ModelLoader.load(model, adapter_path=adapter)
    else:
      _, tokenizer = ModelLoader.load(model)

    input_prompt = _prepare_prompt(prompt, tokenizer, chat_template)

    if compare_base:
      _run_comparison(input_prompt, config)
    else:
      logger.info("Running logit lens pipeline")
      result = LogitProjector.run_logit_lens(loaded_model, tokenizer, input_prompt, config)

      output_str = format_output(result, config.format_type)
      write_output(output_str, config.output_file)

    logger.info("Logit lens analysis complete")

  except ConfigError as e:
    console.print(f"[red]Configuration error: {e}[/red]")
    raise typer.Exit(code=1) from e
  except ModelLoadError as e:
    console.print(f"[red]Failed to load model: {e}[/red]")
    raise typer.Exit(code=1) from e
  except Exception as e:
    console.print(f"[red]Error: {e}[/red]")
    logger.exception("Logit lens failed")
    raise typer.Exit(code=1) from e
  finally:
    if loaded_model is not None:
      try:
        ModelLoader.unload(loaded_model)
      except Exception as e:
        logger.warning(f"Error during cleanup: {e}")


logit_lens_app.command(name="generate")(generate)
