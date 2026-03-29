"""Output formatters (table, JSON, CSV).

Provides strategy pattern formatters for different output types:
- TableFormatter: rich-styled terminal tables
- JsonFormatter: structured JSON with metadata
- CsvFormatter: CSV for spreadsheet import
- get_formatter: factory function to get formatter by type
"""

from mlx_lm_lens.formatters.base import Formatter
from mlx_lm_lens.formatters.csv import CsvFormatter
from mlx_lm_lens.formatters.json import JsonFormatter
from mlx_lm_lens.formatters.registry import get_formatter
from mlx_lm_lens.formatters.table import TableFormatter

__all__ = [
    "Formatter",
    "TableFormatter",
    "JsonFormatter",
    "CsvFormatter",
    "get_formatter",
]
