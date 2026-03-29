"""Configuration dataclasses for all mlx-lm-lens commands.

All parameters flow through these typed config objects, enabling
IDE autocomplete, validation, and type safety. No **kwargs anywhere.
"""

from dataclasses import dataclass, field

from mlx_lm_lens.exceptions import ConfigError


@dataclass
class LogitLensConfig:
    """Configuration for logit-lens command.

    Attributes:
        model_path: Path to MLX model or HuggingFace model ID.
        prompt: Input text to analyze.
        adapter_path: Optional path to LoRA adapter.
        top_k: Number of top tokens to show per layer. Defaults to 5.
        all_positions: If True, analyze all token positions. Defaults to False.
        compare_base: If True, show side-by-side base vs adapted. Defaults to False.
        use_chat_template: If True, apply model's chat template. Defaults to False.
        format_type: Output format ('table', 'json', or 'csv'). Defaults to 'table'.
        output_file: Optional file path for output. Defaults to stdout.
        verbose: Enable verbose logging. Defaults to False.
    """

    model_path: str
    prompt: str
    adapter_path: str | None = None
    top_k: int = 5
    all_positions: bool = False
    compare_base: bool = False
    use_chat_template: bool = False
    format_type: str = "table"
    output_file: str | None = None
    verbose: bool = False

    def validate(self) -> None:
        """Validate configuration.

        Raises:
            ConfigError: If any validation fails.
        """
        if not self.model_path:
            raise ConfigError("model_path is required")
        if not self.prompt:
            raise ConfigError("prompt is required")
        if self.top_k < 1:
            raise ConfigError(f"top_k must be >= 1, got {self.top_k}")
        if self.format_type not in ("table", "json", "csv"):
            raise ConfigError(f"format_type must be table|json|csv, got {self.format_type}")
        if self.compare_base and not self.adapter_path:
            raise ConfigError("compare_base requires adapter_path")


@dataclass
class ActivationAnalysisConfig:
    """Configuration for activations command.

    Attributes:
        model_path: Path to MLX model or HuggingFace model ID.
        reference_model_path: Optional path to reference model for comparison.
        adapter_path: Optional path to LoRA adapter.
        prompt: Input text to analyze.
        metrics: List of metric names to compute.
        batch_prompts_file: Optional JSONL file with multiple prompts.
        compare_base: If True, compare base vs adapted. Defaults to False.
        format_type: Output format ('table', 'json', or 'csv'). Defaults to 'table'.
        output_dir: Optional directory for output files. Defaults to stdout.
        verbose: Enable verbose logging. Defaults to False.
    """

    model_path: str
    prompt: str
    metrics: list[str] = field(default_factory=list)
    reference_model_path: str | None = None
    adapter_path: str | None = None
    batch_prompts_file: str | None = None
    compare_base: bool = False
    format_type: str = "table"
    output_dir: str | None = None
    verbose: bool = False

    def validate(self) -> None:
        """Validate configuration.

        Raises:
            ConfigError: If any validation fails.
        """
        if not self.model_path:
            raise ConfigError("model_path is required")
        if not self.prompt and not self.batch_prompts_file:
            raise ConfigError("Either prompt or batch_prompts_file is required")
        if not self.metrics:
            raise ConfigError("At least one metric is required")
        if self.format_type not in ("table", "json", "csv"):
            raise ConfigError(f"format_type must be table|json|csv, got {self.format_type}")
        if self.compare_base and not self.adapter_path:
            raise ConfigError("compare_base requires adapter_path")


@dataclass
class LogitLensGenerationConfig:
    """Configuration for logit-lens generation mode.

    Generates tokens and logs per-layer predictions for each generated token.

    Attributes:
        model_path: Path to MLX model or HuggingFace model ID.
        prompt: Input text to start generation.
        max_tokens: Number of tokens to generate. Defaults to 100.
        log_from_token: Start logging from this token number (0-indexed). Defaults to 0.
        top_k: Number of top predictions per layer. Defaults to 5.
        temperature: Sampling temperature (1.0=no change, >1=more random, <1=more deterministic). Defaults to 1.0.
        top_p: Nucleus sampling cutoff (0.0-1.0, None=disabled). Defaults to None.
        sampling_method: How to select next token ('greedy', 'top_k', 'nucleus'). Defaults to 'greedy'.
        seed: Random seed for reproducibility. Defaults to None (random).
        include_prompt: Whether to log prompt tokens in output. Defaults to False.
        show_progress: Show progress bar during generation. Defaults to True.
        use_chat_template: Apply model's chat template to prompt. Defaults to True.
        stop_strings: Strings that trigger early termination. Defaults to empty list.
        format_type: Output format ('table', 'json', or 'csv'). Defaults to 'table'.
        output_file: Optional file path for JSON output. Defaults to stdout.
        verbose: Enable verbose logging. Defaults to False.
    """

    model_path: str
    prompt: str
    max_tokens: int = 100
    log_from_token: int = 0
    top_k: int = 5
    temperature: float = 1.0
    top_p: float | None = None
    sampling_method: str = "greedy"
    seed: int | None = None
    include_prompt: bool = False
    show_progress: bool = True
    use_chat_template: bool = True
    stop_strings: list[str] = field(default_factory=list)
    format_type: str = "table"
    output_file: str | None = None
    verbose: bool = False
    drift_correction: bool = False
    drift_threshold: float = 0.3
    drift_baseline_tokens: int = 256
    drift_log: str | None = None
    reference_model_path: str | None = None
    reference_adapter_path: str | None = None

    def validate(self) -> None:
        """Validate configuration.

        Raises:
            ConfigError: If any validation fails.
        """
        if not self.model_path:
            raise ConfigError("model_path is required")
        if not self.prompt:
            raise ConfigError("prompt is required")
        if self.max_tokens < 1:
            raise ConfigError(f"max_tokens must be >= 1, got {self.max_tokens}")
        if self.log_from_token < 0:
            raise ConfigError(f"log_from_token must be >= 0, got {self.log_from_token}")
        if self.log_from_token >= self.max_tokens:
            raise ConfigError(
                f"log_from_token ({self.log_from_token}) must be < max_tokens ({self.max_tokens})"
            )
        if self.top_k < 1:
            raise ConfigError(f"top_k must be >= 1, got {self.top_k}")
        if self.temperature <= 0:
            raise ConfigError(f"temperature must be > 0, got {self.temperature}")
        if self.top_p is not None and (self.top_p <= 0 or self.top_p > 1.0):
            raise ConfigError(f"top_p must be in (0, 1], got {self.top_p}")
        if self.sampling_method not in ("greedy", "top_k", "nucleus"):
            raise ConfigError(
                f"sampling_method must be greedy|top_k|nucleus, got {self.sampling_method}"
            )
        if self.format_type not in ("table", "json", "csv"):
            raise ConfigError(f"format_type must be table|json|csv, got {self.format_type}")
        if any(not s for s in self.stop_strings):
            raise ConfigError("stop_strings must be non-empty")
        if self.drift_threshold <= 0.0 or self.drift_threshold >= 180.0:
            raise ConfigError(
                f"drift_threshold must be in (0, 180) degrees, got {self.drift_threshold}"
            )
        if self.drift_baseline_tokens < 1:
            raise ConfigError(f"drift_baseline_tokens must be >= 1, got {self.drift_baseline_tokens}")
        if (
            self.drift_correction
            and not self.reference_model_path
            and self.drift_baseline_tokens >= self.max_tokens
        ):
            raise ConfigError(
                f"drift_baseline_tokens ({self.drift_baseline_tokens}) must be < max_tokens "
                f"({self.max_tokens}) to allow correction phase"
            )


@dataclass
class CircuitAblationConfig:
    """Configuration for circuit ablation command.

    Attributes:
        model_path: Path to MLX model or HuggingFace model ID.
        prompt: Input text to analyze.
        ablation_method: Method name ('zero', 'mean', 'noise', 'knockout').
        target_token: Optional specific token to measure. Defaults to None.
        layer_range: Optional (start, end) tuple for layer range. Defaults to None.
        format_type: Output format ('table', 'json', or 'csv'). Defaults to 'table'.
        output_dir: Optional directory for output files. Defaults to stdout.
        verbose: Enable verbose logging. Defaults to False.
    """

    model_path: str
    prompt: str
    ablation_method: str
    target_token: str | None = None
    layer_range: tuple[int, int] | None = None
    format_type: str = "table"
    output_dir: str | None = None
    verbose: bool = False

    def validate(self) -> None:
        """Validate configuration.

        Raises:
            ConfigError: If any validation fails.
        """
        if not self.model_path:
            raise ConfigError("model_path is required")
        if not self.prompt:
            raise ConfigError("prompt is required")
        if not self.ablation_method:
            raise ConfigError("ablation_method is required")
        if self.ablation_method not in ("zero", "mean", "noise", "knockout"):
            msg = f"ablation_method must be zero|mean|noise|knockout, got {self.ablation_method}"
            raise ConfigError(msg)
        if self.format_type not in ("table", "json", "csv"):
            raise ConfigError(f"format_type must be table|json|csv, got {self.format_type}")
        if self.layer_range is not None:
            start, end = self.layer_range
            if start >= end:
                raise ConfigError(f"layer_range start must be < end, got ({start}, {end})")
