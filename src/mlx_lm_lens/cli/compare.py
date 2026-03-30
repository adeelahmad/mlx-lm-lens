"""Compare two models side-by-side during generation."""

import json
import logging
import os
from pathlib import Path

import typer
from rich.console import Console
from rich.table import Table

from mlx_lm_lens.config import LogitLensGenerationConfig
from mlx_lm_lens.core.logit_generation import stream_with_logit_lens
from mlx_lm_lens.core.logit_sampling import TokenResult
from mlx_lm_lens.core.model_loader import ModelLoader
from mlx_lm_lens.exceptions import ConfigError, ModelLoadError

logger = logging.getLogger(__name__)
console = Console()


def compare(
    model1: str = typer.Option(
        None,
        "--model1",
        "-m1",
        envvar="Model",
        help="First model path (also reads from $Model env var)",
    ),
    model2: str = typer.Option(
        None,
        "--model2",
        "-m2",
        envvar="Modeln",
        help="Second model path (also reads from $Modeln env var)",
    ),
    prompt_file: str | None = typer.Option(
        None,
        "--prompt-file",
        "-pf",
        envvar="Prompt",
        help="Path to prompt file (also reads from $Prompt env var)",
    ),
    prompt: str | None = typer.Option(
        None,
        "--prompt",
        "-p",
        help="Inline prompt text (alternative to --prompt-file)",
    ),
    max_tokens: int = typer.Option(50, "--max-tokens", "-n", min=1, max=1000),
    top_k: int = typer.Option(5, "--top-k", "-k", min=1),
    top_p: float | None = typer.Option(
        None, "--top-p", min=0.0, max=1.0, help="Nucleus sampling cutoff (for nucleus method)"
    ),
    temperature: float = typer.Option(1.0, "--temperature", "-t", min=0.01),
    sampling_method: str = typer.Option(
        "greedy", "--sampling-method", help="greedy | top_k | nucleus"
    ),
    chat_template: bool = typer.Option(True, "--chat-template/--no-chat-template"),
    output: str | None = typer.Option(None, "--output", "-o", help="Save comparison to JSON"),
) -> None:
  """Compare two models side-by-side.

  Reads model paths from --model1/$Model and --model2/$Modeln,
  prompt from --prompt-file/$Prompt file.

  Generates tokens on both and compares:
  - Generated text (for differences in predictions)
  - Per-layer agreement (how often both models agree on top prediction)
  - Generation speed

  Example:
    export Model=/path/to/model1
    export Modeln=/path/to/model2
    export Prompt=/path/to/prompt.txt
    mlx-lm-lens compare

  Or:
    mlx-lm-lens compare --model1 m1 --model2 m2 --prompt-file p.txt
  """
  if not model1:
    raise ConfigError("model1 required (--model1 or $Model)")
  if not model2:
    raise ConfigError("model2 required (--model2 or $Modeln)")

  # Get prompt from file or inline
  final_prompt = None
  if prompt:
    final_prompt = prompt.strip()
  elif prompt_file:
    if not Path(prompt_file).exists():
      raise ConfigError(f"Prompt file not found: {prompt_file}")
    final_prompt = Path(prompt_file).read_text().strip()
  else:
    raise ConfigError("prompt required (--prompt, --prompt-file, or $Prompt)")

  if not final_prompt:
    raise ConfigError("Prompt is empty")

  console.print(f"[bold cyan]Comparing Models[/bold cyan]")
  console.print(f"[dim]Model 1:[/dim] {model1}")
  console.print(f"[dim]Model 2:[/dim] {model2}")
  console.print(f"[dim]Prompt:[/dim] {final_prompt[:100]}...")
  console.print()

  results_m1: list[TokenResult] = []
  results_m2: list[TokenResult] = []

  # Generate on model 1
  console.print("[bold]Generating on Model 1...[/bold]")
  try:
    model1_obj, tokenizer1 = ModelLoader.load(model1)

    input_prompt1 = final_prompt
    if chat_template and hasattr(tokenizer1, "apply_chat_template"):
      try:
        input_prompt1 = tokenizer1.apply_chat_template(
          [{"role": "user", "content": final_prompt}],
          tokenize=False,
          add_generation_prompt=True,
        )
      except Exception as e:
        logger.warning(f"Chat template failed: {e}")

    prev_len_m1 = 0
    for result in stream_with_logit_lens(
      model1_obj,
      tokenizer1,
      input_prompt1,
      max_tokens=max_tokens,
      log_from_token=0,
      top_k=top_k,
      top_p=top_p,
      temperature=temperature,
      sampling_method=sampling_method,
    ):
      results_m1.append(result)
      # Use properly decoded text, not individual tokens
      new_text = result.generated_text[prev_len_m1:]
      console.print(new_text, end="", highlight=False)
      prev_len_m1 = len(result.generated_text)

    console.print()
    ModelLoader.unload(model1_obj)
  except ModelLoadError as e:
    console.print(f"[red]Error loading model1: {e}[/red]")
    raise

  # Generate on model 2
  console.print()
  console.print("[bold]Generating on Model 2...[/bold]")
  try:
    model2_obj, tokenizer2 = ModelLoader.load(model2)

    input_prompt2 = final_prompt
    if chat_template and hasattr(tokenizer2, "apply_chat_template"):
      try:
        input_prompt2 = tokenizer2.apply_chat_template(
          [{"role": "user", "content": final_prompt}],
          tokenize=False,
          add_generation_prompt=True,
        )
      except Exception as e:
        logger.warning(f"Chat template failed: {e}")

    prev_len_m2 = 0
    for result in stream_with_logit_lens(
      model2_obj,
      tokenizer2,
      input_prompt2,
      max_tokens=max_tokens,
      log_from_token=0,
      top_k=top_k,
      top_p=top_p,
      temperature=temperature,
      sampling_method=sampling_method,
    ):
      results_m2.append(result)
      # Use properly decoded text, not individual tokens
      new_text = result.generated_text[prev_len_m2:]
      console.print(new_text, end="", highlight=False)
      prev_len_m2 = len(result.generated_text)

    console.print()
    ModelLoader.unload(model2_obj)
  except ModelLoadError as e:
    console.print(f"[red]Error loading model2: {e}[/red]")
    raise

  # Compare results
  console.print()
  console.print("[bold]Comparison[/bold]")

  table = Table(title="Model Comparison")
  table.add_column("Metric", style="cyan")
  table.add_column("Model 1", style="magenta")
  table.add_column("Model 2", style="yellow")

  m1_text = results_m1[-1].generated_text if results_m1 else ""
  m2_text = results_m2[-1].generated_text if results_m2 else ""

  table.add_row("Generated Text", m1_text[:50], m2_text[:50])
  table.add_row("Tokens Generated", str(len(results_m1)), str(len(results_m2)))
  table.add_row("Total Time (s)", f"{results_m1[-1].elapsed_sec:.1f}", f"{results_m2[-1].elapsed_sec:.1f}")

  if results_m1 and results_m2:
    speed1 = len(results_m1) / max(results_m1[-1].elapsed_sec, 0.1)
    speed2 = len(results_m2) / max(results_m2[-1].elapsed_sec, 0.1)
    table.add_row("Speed (tok/s)", f"{speed1:.2f}", f"{speed2:.2f}")

  # Layer agreement
  agreement_count = 0
  for i in range(min(len(results_m1), len(results_m2))):
    if results_m1[i].token_id == results_m2[i].token_id:
      agreement_count += 1

  agreement_pct = 100 * agreement_count / max(len(results_m1), len(results_m2))
  table.add_row("Token Agreement", f"{agreement_count}/{len(results_m1)}", f"{agreement_pct:.0f}%")

  console.print(table)

  # Save results
  if output:
    comparison_data = {
      "model1": model1,
      "model2": model2,
      "prompt": final_prompt,
      "config": {
        "max_tokens": max_tokens,
        "top_k": top_k,
        "temperature": temperature,
        "sampling_method": sampling_method,
      },
      "results": {
        "model1": {
          "text": m1_text,
          "tokens": len(results_m1),
          "elapsed_sec": results_m1[-1].elapsed_sec if results_m1 else 0,
          "speed_tok_s": len(results_m1) / max(results_m1[-1].elapsed_sec, 0.1) if results_m1 else 0,
        },
        "model2": {
          "text": m2_text,
          "tokens": len(results_m2),
          "elapsed_sec": results_m2[-1].elapsed_sec if results_m2 else 0,
          "speed_tok_s": len(results_m2) / max(results_m2[-1].elapsed_sec, 0.1) if results_m2 else 0,
        },
      },
      "agreement": {
        "matching_tokens": agreement_count,
        "total_tokens": max(len(results_m1), len(results_m2)),
        "agreement_pct": agreement_pct,
      },
    }

    with open(output, "w") as f:
      json.dump(comparison_data, f, indent=2)
    console.print(f"[green]✓ Comparison saved to {output}[/green]")
