"""Formatter registry for discovering and instantiating formatters.

Provides a registry of available formatters and a factory function to get
formatter instances by name. Raises ConfigError if unknown format requested.
"""

from typing import Any, Type

from mlx_lm_lens.exceptions import ConfigError
from mlx_lm_lens.formatters.base import Formatter
from mlx_lm_lens.formatters.csv import CsvFormatter
from mlx_lm_lens.formatters.json import JsonFormatter
from mlx_lm_lens.formatters.table import TableFormatter

FORMATTER_REGISTRY: dict[str, Type[Any]] = {
    "table": TableFormatter,
    "json": JsonFormatter,
    "csv": CsvFormatter,
}


def get_formatter(format_type: str) -> Formatter:
    """Get formatter instance by type.

    Args:
        format_type: One of "table", "json", or "csv".

    Returns:
        Instantiated formatter object.

    Raises:
        ConfigError: If format_type is not in registry.
    """
    if format_type not in FORMATTER_REGISTRY:
        available = ", ".join(sorted(FORMATTER_REGISTRY.keys()))
        raise ConfigError(
            f"Unknown format: {format_type!r}. Available: {available}"
        )

    formatter_class = FORMATTER_REGISTRY[format_type]
    return formatter_class()
