# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## mlx-lm-lens: Mechanistic Interpretability CLI for MLX Models

Command-line tool for analyzing how transformer models process information layer-by-layer on Apple Silicon (MLX). Provides logit lens, activation analysis (SHAMAN metrics), and circuit discovery via ablation/patching.

## Stack

- **Language**: Python ≥3.10
- **ML Framework**: MLX ≥0.20, mlx-lm ≥0.15
- **CLI**: typer or click (Claude decides)
- **Output**: rich (terminal tables), json, csv
- **Testing**: pytest, hypothesis (property-based tests)
- **Linting**: ruff (strict, zero warnings)
- **Package**: pyproject.toml, pip-installable with `mlx-lm-lens` entry point

## Commands

```bash
# Setup & development
pip install -e .                    # Editable install (required)
ruff check .                        # Lint (strict mode, zero warnings)
pytest tests/ -v                    # Run all tests
pytest tests/ -k logit              # Run specific test
pytest tests/ --cov --cov-report=html  # Coverage report (must be ≥90%)
python -m mlx_lm_lens logit-lens --help  # View CLI help

# Main CLI commands (once implemented)
mlx-lm-lens logit-lens --model <path> --prompt "..."
mlx-lm-lens activations --model <path> --metrics cosine,cka
mlx-lm-lens circuit ablate --model <path> --prompt "..."

# Test with local model (Qwen3-0.6B)
TEST_MODEL_PATH="/Users/adeelahmad/.cache/lm-studio/models/lmstudio-community/Qwen3-0.6B-Base-bf16"
mlx-lm-lens logit-lens --model "$TEST_MODEL_PATH" --prompt "Who are you?"
```

## Architecture

**This is a refactoring project.** The goal is to transform a monolithic 3500+ line legacy script (`legacy-code/activ4.py`) into a clean, modular package following SOLID principles.

**Package structure** (`src/mlx_lm_lens/`):
```
src/mlx_lm_lens/
  __init__.py
  cli/                      # typer/click command handlers
    main.py                 # Entry point and subcommands
    logit_lens.py           # logit-lens command
    activations.py          # activations command
    circuit.py              # circuit discovery subcommands
  core/                     # Core analysis logic (no CLI)
    model_loader.py         # Load base models, LoRA adapters, handle quantization
    activation_capture.py   # Hook into forward pass, capture hidden states
    logit_projection.py     # Project hidden states to vocab space
  metrics/                  # Individual metric modules (plugin architecture)
    base.py                 # Metric interface/protocol
    cosine.py               # Cosine similarity
    cka.py                  # Centered Kernel Alignment
    procrustes.py           # Procrustes residual
    grassmannian.py         # Grassmannian distance
    mad.py                  # Mean Absolute Deviation
    cka.py, rsa.py, etc.    # ... one file per metric
  ablations/                # Ablation methods
    base.py                 # Ablation interface
    zero.py, mean.py, patch.py, etc.
  formatters/               # Output formatting (strategy pattern)
    base.py                 # Formatter interface
    table.py                # Rich terminal tables
    json.py                 # JSON output
    csv.py                  # CSV output
  config.py                 # Dataclasses for all config objects
  exceptions.py             # Custom exception classes
tests/
  conftest.py               # Fixtures (test models, tokenizers)
  unit/                     # Unit tests for each module
  integration/              # Full CLI command tests
  regression/               # Numerical equivalence vs. activ4.py
```

**Key architectural principles**:
1. **Separation of concerns**: Model loading → activation capture → metric computation → formatting → CLI
2. **Plugin architecture for metrics**: Add new metric = create one new file, no changes to existing code
3. **Configuration via dataclasses**: All parameters flow through typed config objects, not raw dicts
4. **Model abstraction**: Analysis code doesn't know if it's running on quantized/full/LoRA models — loader abstracts this
5. **Output formatters as strategies**: Swap JSON for table for CSV without changing analysis code
6. **No global state**: All configuration explicit, no module-level variables with side effects

**Data flow**:
```
CLI args → Config dataclass → ModelLoader (returns abstract model interface)
→ ActivationCapture (hooks forward pass, yields per-layer hidden states)
→ MetricComputation (each metric as independent function)
→ Formatter (table/json/csv strategy)
→ stdout or file
```

## Mandatory Constraints (Non-Negotiable)

These come from `PROJECT.md` and **OVERRIDE** normal development practices:

1. **File size limit**: No file >200 lines (excluding comments/blank lines). Break into smaller modules.
2. **Test coverage**: ≥90% across all modules. Every public function, every CLI command, every edge case must have tests.
3. **SOLID principles**: Mandatory. Every class/module has ONE clear responsibility.
4. **Clean Code**: Small functions (≤20 lines preferred), meaningful names, no magic numbers, no dead code.
5. **Type hints**: On every function signature. No `Any` unless absolutely unavoidable. Use `Protocol` classes for dependency injection.
6. **Docstrings**: Google-style format on every public class and function.
7. **No global mutable state**: All config flows through explicit dependency injection.
8. **Error handling**: No bare `except`. Specific errors with meaningful messages. Custom exception classes.
9. **Logging over print()**: Use Python `logging` module with configurable levels. No `print()` for user output — use `rich` console.
10. **Zero ruff warnings**: Strict linter mode. Every warning must be fixed.
11. **Publication-ready**: README, LICENSE (Apache 2.0), pyproject.toml, proper package structure.
12. **Immediately executable**: CLI must work with `mlx-lm-lens <command>` right after install.

**If constraints conflict with speed, the constraint wins. Quality over speed. Always.**

## Current State

The codebase is **pre-implementation**. What exists:
- `PROJECT.md` — complete specification with all features, requirements, and constraints
- `legacy-code/activ4.py` — 3500+ line monolithic script with known bugs (see `legacy-code/activ4_review.md`)
- `skills/` — external documentation (MLX, PyTorch, model loading patterns)

What does NOT exist yet:
- `src/mlx_lm_lens/` package structure
- CLI commands
- Tests
- pyproject.toml setup

## Implementation Roadmap

This follows the phases in `PROJECT.md`:

**Phase 1: Refactor & Foundation** (must complete before other work)
- Read entire `legacy-code/activ4.py` and understand every function
- Design package architecture (modules, interfaces, protocols)
- Refactor SHAMAN metrics into individual files with tests
- Build model loader abstraction
- Build output formatter system
- Verify numerical equivalence with legacy code (regression tests)

**Phase 2: Logit Lens**
- Forward pass with hidden state capture
- Logit projection pipeline
- Flip-point detection
- Comparison mode (base vs adapted)
- CLI wiring
- Full test coverage

**Phase 3: Activation Analysis**
- Wire refactored metrics into CLI
- Batch processing from JSONL
- Anomaly detection
- Statistical aggregation
- Tests

**Phase 4: Circuit Discovery**
- Ablation methods (zero, mean, noise, knockout)
- Activation patching
- Residual stream decomposition
- Weight angle computation
- CLI wiring
- Tests

**Phase 5: Polish & Publish**
- README with installation and examples
- GitHub Actions CI (lint, test, coverage)
- Example notebooks
- Final coverage check (≥90%)

## Testing Strategy (Required)

From `PROJECT.md` — non-negotiable:

- **Unit tests**: Every metric function, every formatter, every loader function
- **Integration tests**: Full CLI command execution with small test models
- **Regression tests**: Numerical equivalence between refactored code and `activ4.py` on same inputs
- **Fixture models**: Use tiny 2-layer transformer for fast test execution
- **Snapshot tests**: CLI output formatting (table/JSON/CSV) matches expected
- **Property-based tests**: Metric properties (e.g., cosine(A,A)=1.0, CKA is symmetric)

**Coverage must be ≥90%** across all modules. Check with:
```bash
pytest tests/ --cov=src/mlx_lm_lens --cov-report=term-missing
```

## Key Files to Understand

**Before writing any new code, read these in order**:

1. `PROJECT.md` — Complete specification, all requirements, all constraints
2. `legacy-code/activ4_review.md` — What's broken in the legacy code (avoid these bugs)
3. `legacy-code/activ4.py` — The actual monolithic implementation (understand every function)

## Common Development Tasks

**Starting Phase 1 (refactoring)**:
1. Create `src/mlx_lm_lens/` package structure with `__init__.py`
2. Create `pyproject.toml` with dependencies and entry point
3. Read and map every function in `activ4.py` to the new architecture
4. Create metric modules one at a time, with corresponding unit tests
5. Add regression tests comparing output to `activ4.py`

**Adding a new CLI command**:
1. Design the command function in `cli/module.py` with typer decorators
2. Create corresponding test in `tests/integration/test_module.py`
3. Wire in any new config dataclasses to `config.py`
4. Ensure ≥90% coverage for your new code

**Adding a new metric**:
1. Create `metrics/newmetric.py` implementing the Metric protocol
2. Add unit tests in `tests/unit/metrics/test_newmetric.py`
3. Register metric in metric registry (avoid hardcoded if/else)
4. No changes to existing metric files or CLI code

**Running tests**:
```bash
pytest tests/ -v              # Verbose mode
pytest tests/ -k "metric"     # Run tests matching "metric"
pytest --cov --cov-report=html  # HTML coverage report
```

## Important Notes

- **No MLX-specific hacks**: All MLX model loading should use `mlx_lm` library patterns
- **LoRA support**: Must work transparently with quantized + LoRA combos
- **Memory management**: Apple Silicon has unified memory — don't leak activation tensors between phases
- **Lazy evaluation**: MLX is lazy — may need `mx.eval()` calls to force computation at checkpoints
- **No hardcoded paths**: Never hardcode model paths, adapter paths, or file paths
- **Error messages are user-facing**: Custom exceptions with clear, actionable messages
- **Logging not printing**: All debugging/info via `logging` module, set levels with `--verbose`/`--quiet`

## Testing Environment

**Local test model (for manual testing + fixtures):**
```bash
# Qwen3-0.6B (0.6B params, bf16, fast for testing)
export TEST_MODEL_PATH="/Users/adeelahmad/.cache/lm-studio/models/lmstudio-community/Qwen3-0.6B-Base-bf16"

# Verify model exists
ls "$TEST_MODEL_PATH"

# Use in tests
pytest tests/ -v  # Fixtures should use this path for integration tests
```

**For unit tests**, use tiny synthetic models (see `tests/conftest.py`). For integration tests, use the local Qwen3-0.6B model above.

## When to Create Tasks/Plan

For non-trivial work:
1. Use EnterPlanMode to design the implementation before coding
2. Create TaskCreate to break Phase X work into discrete steps
3. Use code-reviewer and qa subagents after writing code (see workflow in `~/.claude/CLAUDE.md`)

## Resources

- `PROJECT.md` — Single source of truth for all requirements
- `PLAN.md` — Detailed implementation roadmap with all requirements (60+ items)
- `legacy-code/activ4_review.md` — Known bugs to avoid
- `legacy-code/activ4.py` — Reference implementation (read carefully before refactoring)
- MLX docs: https://ml-explore.github.io/mlx/build/html/
- mlx-lm docs: https://github.com/ml-explore/mlx-lm
