"""Logit-lens generate command with streaming TUI."""

import json
import logging
import signal

import mlx.core as mx
import typer
from rich.console import Console
from rich.live import Live

from mlx_lm_lens.cli.generate_tui import GenerateTUI
from mlx_lm_lens.cli.wordcloud_renderer import render_wordcloud
from mlx_lm_lens.config import LogitLensGenerationConfig
from mlx_lm_lens.core.activation_capture import ActivationCapture
from mlx_lm_lens.core.drift_correction import DriftCorrector
from mlx_lm_lens.core.logit_generation import stream_with_logit_lens
from mlx_lm_lens.core.logit_sampling import TokenResult
from mlx_lm_lens.core.model_loader import ModelLoader
from mlx_lm_lens.exceptions import ConfigError, ModelLoadError

logger = logging.getLogger(__name__)
console = Console()


def generate(
  model: str = typer.Option(..., "--model", "-m", help="Path to MLX model"),
  prompt: str = typer.Option(..., "--prompt", "-p", help="Initial prompt"),
  max_tokens: int = typer.Option(100, "--max-tokens", "-n", min=1, max=1000),
  log_from_token: int = typer.Option(0, "--log-from-token", "-l", min=0),
  top_k: int = typer.Option(5, "--top-k", "-k", min=1),
  temperature: float = typer.Option(1.0, "--temperature", "-t", min=0.01),
  top_p: float | None = typer.Option(None, "--top-p"),
  sampling_method: str = typer.Option("greedy", "--sampling-method"),
  seed: int | None = typer.Option(None, "--seed", "-s"),
  include_prompt: bool = typer.Option(False, "--include-prompt"),
  show_progress: bool = typer.Option(True, "--show-progress/--no-progress"),
  chat_template: bool = typer.Option(True, "--chat-template/--no-chat-template"),
  stop: list[str] = typer.Option(None, "--stop"),  # noqa: B008
  no_tui: bool = typer.Option(False, "--no-tui", help="Disable TUI"),
  wordcloud: bool = typer.Option(False, "--wordcloud", help="Show word cloud"),
  wordcloud_out: str | None = typer.Option(None, "--wordcloud-out"),
  format_type: str = typer.Option("table", "--format", "-f"),
  output: str | None = typer.Option(None, "--output", "-o"),
  drift_correction: bool = typer.Option(
    False, "--drift-correction", help="Enable per-layer drift correction"
  ),
  drift_threshold: float = typer.Option(
    0.3,
    "--drift-threshold",
    min=0.001,
    max=179.9,
    help="Angle threshold in degrees",
  ),
  drift_baseline_tokens: int = typer.Option(
    256, "--drift-baseline-tokens", min=1, help="Tokens to build baseline from"
  ),
  drift_log: str | None = typer.Option(
    None, "--drift-log", help="JSONL file for per-layer angle log"
  ),
  reference_model: str | None = typer.Option(
    None, "--reference-model", help="Reference model path for drift baseline"
  ),
  reference_adapter: str | None = typer.Option(
    None, "--reference-adapter", help="Reference adapter path"
  ),
) -> None:
  """Generate tokens and log per-layer predictions with live TUI.

  Example:
    mlx-lm-lens logit-lens generate --model qwen3 --prompt "hello" \\
      --max-tokens 100 --stop "." --wordcloud
  """
  loaded_model = None
  partial_results: list[TokenResult] = []
  interrupted = False
  stop = stop or []

  def _sigint_handler(sig: int, frame) -> None:
    nonlocal interrupted
    interrupted = True

  signal.signal(signal.SIGINT, _sigint_handler)

  try:
    config = LogitLensGenerationConfig(
      model_path=model,
      prompt=prompt,
      max_tokens=max_tokens,
      log_from_token=log_from_token,
      top_k=top_k,
      temperature=temperature,
      top_p=top_p,
      sampling_method=sampling_method,
      seed=seed,
      include_prompt=include_prompt,
      show_progress=show_progress,
      use_chat_template=chat_template,
      stop_strings=stop,
      format_type=format_type,
      output_file=output,
      drift_correction=drift_correction,
      drift_threshold=drift_threshold,
      drift_baseline_tokens=drift_baseline_tokens,
      drift_log=drift_log,
      reference_model_path=reference_model,
      reference_adapter_path=reference_adapter,
    )
    config.validate()

    logger.info(f"Loading model: {model}")
    loaded_model, tokenizer = ModelLoader.load(model)

    input_prompt = prompt
    if chat_template and hasattr(tokenizer, "apply_chat_template"):
      try:
        input_prompt = tokenizer.apply_chat_template(
          [{"role": "user", "content": prompt}],
          tokenize=False,
        )
        logger.info("Applied chat template to prompt")
      except (AttributeError, TypeError, RuntimeError) as e:
        logger.warning(f"Chat template failed, using raw prompt: {e}")

    logger.info(
      f"Generating {max_tokens} tokens (method={sampling_method}, temp={temperature})"
    )

    tui = None if no_tui else GenerateTUI(model, input_prompt[:50], max_tokens, sampling_method)

    if no_tui:
      console.print("[bold cyan]Generating tokens...[/bold cyan]")

    live_ctx = None
    if tui:
      live_ctx = Live(tui.get_layout(), refresh_per_second=8, console=console)
      live_ctx.__enter__()

    corrector = None
    if config.drift_correction:
      corrector = DriftCorrector(
        threshold_deg=config.drift_threshold,
        baseline_tokens=config.drift_baseline_tokens,
        log_path=config.drift_log,
      )

      # Load reference model and compute baselines if provided
      if config.reference_model_path:
        logger.info(f"Loading reference model: {config.reference_model_path}")
        ref_model, _ = ModelLoader.load(
          config.reference_model_path, adapter_path=config.reference_adapter_path
        )

        try:
          # Run prompt through reference model to get activations
          ref_prompt_ids = tokenizer.encode(input_prompt)
          if not isinstance(ref_prompt_ids, list):
            ref_prompt_ids = ref_prompt_ids.tolist()

          ref_activations = ActivationCapture.forward_with_capture(
            ref_model, mx.array([ref_prompt_ids])
          )

          # Extract normalized last-position vectors as baselines
          baselines: dict[int, mx.array] = {}
          for key in sorted(ref_activations.keys()):
            if not key.startswith("layer_"):
              continue

            hidden = ref_activations[key]
            if hidden.ndim == 3:
              h_last = hidden[0, -1, :]
            else:
              h_last = hidden[-1, :]

            # L2 normalize
            h_norm = mx.sqrt(mx.sum(h_last * h_last))
            if float(h_norm) < 1e-8:
              continue

            h_unit = h_last / (h_norm + 1e-8)
            mx.eval(h_unit)
            layer_idx = int(key.split("_")[1])
            baselines[layer_idx] = h_unit

          corrector.set_reference_baselines(baselines)
          logger.info(f"Reference baselines set for {len(baselines)} layers")
        finally:
          ModelLoader.unload(ref_model)

    prev_text_len = 0
    try:
      for result in stream_with_logit_lens(
        loaded_model,
        tokenizer,
        input_prompt,
        max_tokens=max_tokens,
        log_from_token=log_from_token,
        top_k=top_k,
        temperature=temperature,
        top_p=top_p,
        sampling_method=sampling_method,
        seed=seed,
        stop_strings=stop or None,
        drift_corrector=corrector,
      ):
        partial_results.append(result)

        if tui:
          tui.update(result)
          live_ctx.update(tui.get_layout())
        else:
          new_text = result.generated_text[prev_text_len:]
          console.print(new_text, end="", highlight=False)
          prev_text_len = len(result.generated_text)

        if interrupted:
          console.print("\n[yellow]Interrupted — saving partial results...[/yellow]")
          break

    finally:
      if live_ctx:
        live_ctx.__exit__(None, None, None)

    if no_tui:
      console.print("\n")

    _save_results(partial_results, config, input_prompt, tokenizer, output)

    if wordcloud and partial_results:
      if tui:
        freq = tui.build_token_frequency(partial_results)
      else:
        freq = _build_token_frequency(partial_results)
      render_wordcloud(freq, wordcloud_out, console)

    logger.info("Generation complete")

  except ConfigError as e:
    console.print(f"[red]Configuration error: {e}[/red]")
    raise typer.Exit(code=1) from e
  except ModelLoadError as e:
    console.print(f"[red]Failed to load model: {e}[/red]")
    raise typer.Exit(code=1) from e
  except Exception as e:
    console.print(f"[red]Error: {e}[/red]")
    logger.exception("Generation failed")
    raise typer.Exit(code=1) from e
  finally:
    signal.signal(signal.SIGINT, signal.SIG_DFL)
    if loaded_model is not None:
      try:
        ModelLoader.unload(loaded_model)
      except Exception as e:
        logger.warning(f"Error during cleanup: {e}")


def _save_results(
  partial_results: list[TokenResult],
  config: LogitLensGenerationConfig,
  prompt: str,
  tokenizer,
  output_path: str | None,
) -> None:
  """Save partial results to JSON."""
  if not output_path or not partial_results:
    return

  generated_tokens = [r.token_str for r in partial_results]
  token_predictions = [
    {
      "token_number": r.token_num,
      "generated_token": r.token_str,
      "generated_token_id": r.token_id,
      "layer_predictions": r.layer_predictions,
    }
    for r in partial_results
  ]

  result_dict = {
    "prompt": prompt,
    "generated_tokens": generated_tokens,
    "token_predictions": token_predictions,
    "config": {
      "max_tokens": config.max_tokens,
      "log_from_token": config.log_from_token,
      "top_k": config.top_k,
      "temperature": config.temperature,
      "top_p": config.top_p,
      "sampling_method": config.sampling_method,
      "logged_tokens": len(token_predictions),
    },
  }

  with open(output_path, "w") as f:
    json.dump(result_dict, f, indent=2)
  console.print(f"[green]✓ Results saved to {output_path}[/green]")


def _build_token_frequency(results: list[TokenResult]) -> dict:
  """Build token frequency from all results."""
  freq = {}
  for r in results:
    for layer_preds in r.layer_predictions.values():
      for pred in layer_preds:
        token = pred["token"]
        prob = pred["probability"]
        freq[token] = freq.get(token, 0) + prob
  return freq
