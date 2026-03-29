"""Custom exception classes for mlx-lm-lens.

All exceptions inherit from MLXLensError (the base). Specific exceptions
are used for domain-specific error conditions to enable targeted error handling
and user-friendly error messages.
"""


class MLXLensError(Exception):
    """Base exception for all mlx-lm-lens errors.

    This is the root exception class for the package. All domain-specific
    exceptions inherit from this class. Useful for catching any mlx-lm-lens
    error without catching unrelated exceptions.
    """


class ModelLoadError(MLXLensError):
    """Raised when model or adapter loading fails.

    This exception is raised when:
    - Model file not found or inaccessible
    - Model format not supported
    - LoRA adapter file not found or malformed
    - Adapter fusion with base model fails
    - Quantization loading fails
    - Memory allocation fails during model loading
    """


class ActivationCaptureError(MLXLensError):
    """Raised when activation capture during forward pass fails.

    This exception is raised when:
    - Forward pass execution fails
    - Hidden state extraction from layers fails
    - Hook registration fails
    - Hook callback encounters an error
    - Tensor shape mismatches during capture
    - Memory errors while storing activations
    """


class MetricComputationError(MLXLensError):
    """Raised when metric computation fails.

    This exception is raised when:
    - Input arrays have invalid shapes
    - Numerical computation produces NaN or infinity
    - Required dependencies (scipy, sklearn) are missing
    - Matrix inversion or decomposition fails
    - Metric-specific constraints are violated
    """


class AblationError(MLXLensError):
    """Raised when ablation or circuit discovery fails.

    This exception is raised when:
    - Ablation method initialization fails
    - Layer modification during ablation fails
    - Model state restoration fails
    - Patching between models fails
    - Layer range specification is invalid
    """


class DriftCorrectionError(MLXLensError):
    """Raised when geometric drift correction fails.

    This exception is raised when:
    - Baseline accumulation produces degenerate (zero-norm) directions
    - Hidden state shape is incompatible with stored baseline shape
    - Log file cannot be opened or written
    - Numerical computation produces NaN or infinity during correction
    """


class FormattingError(MLXLensError):
    """Raised when output formatting fails.

    This exception is raised when:
    - Data structure doesn't match formatter expectations
    - File I/O fails during output
    - Invalid format type specified
    - Data contains values that can't be serialized
    """


class ConfigError(MLXLensError):
    """Raised when configuration validation fails.

    This exception is raised when:
    - Required config fields are missing
    - Config field values are invalid (wrong type, out of range)
    - Metric or ablation method names don't exist
    - File paths in config are invalid
    - Incompatible config combinations are specified
    """
