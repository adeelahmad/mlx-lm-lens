"""Run activation analysis with metrics for single and batch prompts."""

import json
import logging
from typing import Any, Dict, List

import mlx.core as mx
import numpy as np

from mlx_lm_lens.core.activation_capture import ActivationCapture
from mlx_lm_lens.core.model_loader import ModelLoader
from mlx_lm_lens.metrics.base import MetricComputationError, get_metric

logger = logging.getLogger(__name__)


class ActivationAnalyzer:
    """Compute activation metrics for single and batch prompts."""

    @staticmethod
    def run_single_prompt(
        model: Any,
        tokenizer: Any,
        prompt: str,
        metrics_list: List[str],
    ) -> Dict[str, Dict[str, float]]:
        """Run activation analysis on single prompt.

        Args:
            model: MLX model with forward pass.
            tokenizer: Tokenizer for encoding.
            prompt: Input text.
            metrics_list: Metric names to compute.

        Returns:
            Dict[layer_key][metric_name] = float.
        """
        try:
            token_ids = tokenizer.encode(prompt)
            input_ids = mx.array(token_ids)
            activations = ActivationCapture.forward_with_capture(
                model, input_ids
            )

            np_activations = {
                k: np.array(v) for k, v in activations.items()
                if isinstance(v, mx.array)
            }

            results: Dict[str, Dict[str, float]] = {}
            for layer_key, activation in np_activations.items():
                results[layer_key] = {}
                for metric_name in metrics_list:
                    try:
                        metric = get_metric(metric_name)
                        val = metric.compute(activation, activation)
                        results[layer_key][metric_name] = float(val)
                    except MetricComputationError as e:
                        logger.warning(f"{layer_key}/{metric_name}: {e}")
                        results[layer_key][metric_name] = float("nan")
            return results

        except Exception as e:
            logger.error(f"Single-prompt analysis failed: {e}")
            raise

    @staticmethod
    def run_batch_prompts(
        model_a_path: str,
        model_b_path: str,
        prompts_jsonl: str,
        metrics_list: List[str],
    ) -> Dict[str, Any]:
        """Run batch activation analysis with aggregation.

        Args:
            model_a_path: Path to first model.
            model_b_path: Path to second model.
            prompts_jsonl: Path to JSONL file.
            metrics_list: Metrics to compute.

        Returns:
            Dict with "per_prompt" and "aggregated" keys.
        """
        model_a, tokenizer_a = ModelLoader.load(model_a_path)
        model_b, tokenizer_b = ModelLoader.load(model_b_path)

        try:
            per_prompt_results = []
            layer_metrics: Dict[str, Dict[str, List[float]]] = {}

            with open(prompts_jsonl) as f:
                for line in f:
                    if not line.strip():
                        continue
                    data = json.loads(line)
                    prompt = data.get("prompt", "")
                    metadata = data.get("metadata", {})

                    result_a = ActivationAnalyzer.run_single_prompt(
                        model_a, tokenizer_a, prompt, metrics_list
                    )
                    result_b = ActivationAnalyzer.run_single_prompt(
                        model_b, tokenizer_b, prompt, metrics_list
                    )

                    diff = _compute_diff(result_a, result_b, metrics_list)
                    per_prompt_results.append({
                        "prompt": prompt,
                        "metadata": metadata,
                        "model_a": result_a,
                        "model_b": result_b,
                        "diff": diff,
                    })

                    for layer, metrics_dict in diff.items():
                        if layer not in layer_metrics:
                            layer_metrics[layer] = {}
                        for metric, value in metrics_dict.items():
                            if metric not in layer_metrics[layer]:
                                layer_metrics[layer][metric] = []
                            if np.isfinite(value):
                                layer_metrics[layer][metric].append(value)

            aggregated = _compute_agg(layer_metrics)
            return {
                "per_prompt": per_prompt_results,
                "aggregated": aggregated,
            }

        finally:
            ModelLoader.unload(model_a)
            ModelLoader.unload(model_b)


def _compute_diff(
    result_a: Dict[str, Dict[str, float]],
    result_b: Dict[str, Dict[str, float]],
    metrics_list: List[str],
) -> Dict[str, Dict[str, float]]:
    """Compute metric differences between two results."""
    diffs: Dict[str, Dict[str, float]] = {}
    for layer_key in result_a:
        diffs[layer_key] = {}
        for metric_name in metrics_list:
            val_a = result_a[layer_key].get(metric_name, float("nan"))
            val_b = result_b[layer_key].get(metric_name, float("nan"))
            if np.isfinite(val_a) and np.isfinite(val_b):
                diffs[layer_key][metric_name] = float(val_a - val_b)
            else:
                diffs[layer_key][metric_name] = float("nan")
    return diffs


def _compute_agg(
    layer_metrics: Dict[str, Dict[str, List[float]]],
) -> Dict[str, Dict[str, Dict[str, float]]]:
    """Compute mean/std/min/max for each layer-metric."""
    aggregated: Dict[str, Dict[str, Dict[str, float]]] = {}
    for layer_key, metrics_dict in layer_metrics.items():
        aggregated[layer_key] = {}
        for metric_name, values in metrics_dict.items():
            if values:
                aggregated[layer_key][metric_name] = {
                    "mean": float(np.mean(values)),
                    "std": float(np.std(values)),
                    "min": float(np.min(values)),
                    "max": float(np.max(values)),
                }
    return aggregated
