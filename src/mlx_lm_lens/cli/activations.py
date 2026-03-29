"""CLI command for activation analysis."""

import logging
from pathlib import Path
from typing import List, Optional

import typer

from mlx_lm_lens.analysis.anomaly import AnomalyDetector
from mlx_lm_lens.core.model_loader import ModelLoader
from mlx_lm_lens.formatters.registry import get_formatter
from mlx_lm_lens.metrics.base import list_metrics
from mlx_lm_lens.runner.activation_runner import ActivationAnalyzer
from mlx_lm_lens.runner.shaman_runner import SHAMANValidator

logger = logging.getLogger(__name__)

activations_app = typer.Typer(
    help="Analyze activations across model layers"
)


@activations_app.command()
def activations(
    model: str = typer.Option(..., "--model", "-m"),
    reference: Optional[str] = typer.Option(None, "--reference", "-r"),
    adapter: Optional[str] = typer.Option(None, "--adapter", "-a"),
    prompt: Optional[str] = typer.Option(None, "--prompt", "-p"),
    prompts_file: Optional[str] = typer.Option(
        None, "--prompts-file"
    ),
    metrics: str = typer.Option("cosine,cka", "--metrics"),
    compare_base: bool = typer.Option(False, "--compare-base"),
    format_type: str = typer.Option("table", "--format", "-f"),
    output: Optional[str] = typer.Option(None, "--output", "-o"),
    detect_anomalies: bool = typer.Option(False, "--anomalies"),
    validate_hypotheses: bool = typer.Option(
        False, "--validate-hypotheses"
    ),
) -> None:
    """Analyze hidden state activations across model layers."""
    try:
        if not prompt and not prompts_file:
            raise ValueError("Must specify --prompt or --prompts-file")
        if prompt and prompts_file:
            raise ValueError(
                "Cannot specify both --prompt and --prompts-file"
            )
        if compare_base and not adapter:
            raise ValueError("--compare-base requires --adapter")

        metrics_list = (
            list_metrics()
            if metrics.lower() == "all"
            else [m.strip() for m in metrics.split(",")]
        )
        _validate_metrics(metrics_list)

        model_b = reference or (adapter if compare_base else model)

        if prompt:
            _run_single(model, model_b, prompt, metrics_list,
                       format_type, output, detect_anomalies,
                       validate_hypotheses)
        else:
            _run_batch(model, model_b, prompts_file, metrics_list,
                      format_type, output, detect_anomalies)

    except Exception as e:
        typer.secho(f"Error: {e}", fg=typer.colors.RED, err=True)
        raise typer.Exit(1)


def _validate_metrics(metrics_list: List[str]) -> None:
    """Validate metric names."""
    available = set(list_metrics())
    for m in metrics_list:
        if m not in available:
            raise ValueError(f"Unknown metric '{m}'")


def _run_single(
    model_a: str, model_b: str, prompt: str, metrics_list: List[str],
    format_type: str, output: Optional[str], detect_anomalies: bool,
    validate_hypotheses: bool,
) -> None:
    """Run single-prompt analysis."""
    model_a_obj, tokenizer_a = ModelLoader.load(model_a)
    model_b_obj, tokenizer_b = ModelLoader.load(model_b)

    try:
        result_a = ActivationAnalyzer.run_single_prompt(
            model_a_obj, tokenizer_a, prompt, metrics_list
        )
        result_b = ActivationAnalyzer.run_single_prompt(
            model_b_obj, tokenizer_b, prompt, metrics_list
        )

        output_data = {
            "prompt": prompt,
            "model_a": result_a,
            "model_b": result_b,
        }

        if detect_anomalies:
            output_data["anomalies"] = (
                AnomalyDetector.detect_layer_anomalies(result_a)
            )

        if validate_hypotheses:
            output_data["hypotheses"] = (
                SHAMANValidator.validate_all_hypotheses(result_a)
            )

        formatter = get_formatter(format_type)
        formatted = formatter.format(output_data)

        if output:
            Path(output).write_text(formatted)
            typer.secho(f"Saved to {output}", fg=typer.colors.GREEN)
        else:
            typer.echo(formatted)

    finally:
        ModelLoader.unload(model_a_obj)
        ModelLoader.unload(model_b_obj)


def _run_batch(
    model_a: str, model_b: str, prompts_file: str,
    metrics_list: List[str], format_type: str, output: Optional[str],
    detect_anomalies: bool,
) -> None:
    """Run batch analysis."""
    result = ActivationAnalyzer.run_batch_prompts(
        model_a, model_b, prompts_file, metrics_list
    )

    if detect_anomalies and "aggregated" in result:
        result["anomalies"] = AnomalyDetector.detect_layer_anomalies(
            result["aggregated"]
        )

    formatter = get_formatter(format_type)
    formatted = formatter.format(result)

    if output:
        Path(output).write_text(formatted)
        typer.secho(f"Saved to {output}", fg=typer.colors.GREEN)
    else:
        typer.echo(formatted)
