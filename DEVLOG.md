# Project Dev Log

## Working State

**Session:** 8 | **Date:** 2026-03-29

### Active Task
Phase 4b: Circuit discovery CLI commands wiring (ablate, patch, decompose, angles)

- [x] Step 1: Create AblationSweep class (sweep_all_layers, rank_layers)
- [x] Step 2: Create ResidualStreamDecomposer (decompose_residual, show_contributions)
- [x] Step 3: Create WeightAngleAnalyzer (compute_weight_angles, compute_cumulative_rotation)
- [x] Step 4: Create CLI circuit subcommands (4 separate files: circuit_ablate, circuit_patch, circuit_decompose, circuit_angles)
- [x] Step 5: Wire circuit_app into main.py
- [x] Step 6: Add load_tokenizer function to tokenizer_utils
- [x] Step 7: Create integration tests for circuit CLI
- [ ] Step 8: Run pytest and ruff checks

### Key Files (current shape)

**`src/mlx_lm_lens/ablations/base.py`** (28 lines, NEW)
AblationMethod Protocol: name property, apply() method. Defines interface for all ablation implementations.

**`src/mlx_lm_lens/ablations/zero.py`** (33 lines, NEW)
ZeroAblation: returns mx.zeros_like(layer_output). Measures total contribution of layer.

**`src/mlx_lm_lens/ablations/mean.py`** (42 lines, NEW)
MeanAblation: computes mean across seq dimension, broadcasts to full shape. Measures deviation from baseline.

**`src/mlx_lm_lens/ablations/noise.py`** (52 lines, NEW)
NoiseAblation: adds Gaussian noise with std proportional to output std. Scales via noise_scale parameter.

**`src/mlx_lm_lens/ablations/knockout.py`** (47 lines, NEW)
ResidualKnockout: returns layer_input (skip connection) from metadata, removing residual addition.

**`src/mlx_lm_lens/ablations/registry.py`** (41 lines, NEW)
ABLATION_REGISTRY dict, get_ablation(name, **kwargs) factory with error handling. Plugin pattern.

**`src/mlx_lm_lens/circuit/tracer.py`** (180 lines, NEW)
HeadImportanceTrace dataclass (importances, norms, entropy). CircuitTracer.compute_head_importance() extracts per-head metrics from attention module (V-norms + entropy). run_circuit_forward() captures traces at each layer.

**`src/mlx_lm_lens/circuit/ablation.py`** (173 lines, NEW)
AblationSweep class: _compute_kl_divergence() numerically stable KL (scipy.special.softmax). sweep_all_layers() ablates each layer, measures KL divergence. rank_layers() sorts by importance.

**`src/mlx_lm_lens/circuit/patch.py`** (187 lines, NEW)
ActivationPatcher: _compute_kl_divergence() stable KL. patch_layer() replaces layer_i in model_A with model_B's. sweep_patch_all_layers() tests all layers.

**`tests/unit/ablations/test_ablations.py`** (156 lines, NEW)
15 tests: shape preservation, name properties, broadcasting (mean), perturbation (noise), knockout metadata validation, registry instantiation, kwargs handling.

**`tests/integration/test_circuit.py`** (105 lines, NEW)
7 tests: head importance computation, layer ranking, KL divergence (identical/different), activation patcher KL.

### Decisions (active)

- Ablation interface: Protocol vs ABC — chose Protocol for minimal coupling, follows project patterns
- KL divergence: scipy.special.softmax for numerical stability, clips to [0, 1e6] to prevent inf
- Head importance: combines V-projection norms + attention entropy with broadcasting for GQA
- Error handling: graceful fallback to uniform importance if head computation fails
- Test coverage: unit tests for ablation methods, integration tests for KL + ranking

### Next Steps

1. Spawn code-reviewer subagent on Phase 4a files
2. Spawn qa subagent (generate + run tests)
3. Fix issues from both reports
4. Verify line counts: all <200 ✓
5. Run full test suite (unit + integration)
6. Final ruff strict check

### Blockers

None — Phase 2 implementation complete, ready for code-reviewer and qa subagents.

### Watch Out

- Logit lens assumes lm_head exists on model — should fail gracefully if missing (currently logs warning)
- Numerically stable softmax critical: without subtracting max, exp() can overflow on large logits
- Flip detection compares strings — relies on tokenizer.decode() being consistent
- Chat template application can fail silently (logs warning, continues with raw prompt)
- Config validation must happen before model loading to fail fast on bad params

---

## Session Archive

### Session 8 -- 2026-03-29: Phase 5 - Polish & Publish
**What we did:** Complete user-facing documentation, CI/CD setup, example scripts. Created README (450 lines, 3 use cases), Apache 2.0 LICENSE, GitHub Actions workflow (lint/test/build), 3 example scripts, CONTRIBUTING.md, tests/README.md, updated __init__.py with public API.
**Files:** README.md, LICENSE, .github/workflows/ci.yml, examples/{logit_lens,activation_analysis,circuit_discovery}_example.py, CONTRIBUTING.md, tests/README.md, src/mlx_lm_lens/__init__.py
**Decisions:** README emphasizes Apple Silicon + mechanistic interpretability, CI fails on ruff warnings + coverage <90%, examples runnable pseudocode with real imports, documentation focused on "why" not just "what"

### Session 7 -- 2026-03-29: Phase 3 - Activation Analysis feature
**What we did:** Complete activation analysis: ActivationAnalyzer (single/batch), AnomalyDetector (rolling window z-scores), SHAMANValidator (all 15 hypotheses), CLI command, 26 integration tests.
**Files:** src/mlx_lm_lens/runner/{activation_runner.py,shaman_runner.py}, src/mlx_lm_lens/analysis/anomaly.py, src/mlx_lm_lens/cli/activations.py, tests/integration/test_activations.py, tests/fixtures/sample_prompts.jsonl
**Decisions:** Batch aggregates to mean/std/min/max per layer, rolling window anomaly detection, metrics validation before analysis, all files <200 lines

### Session 5 -- 2026-03-29: Phase 2 - Logit Lens feature
**What we did:** Implemented complete logit lens pipeline: LogitProjector class, CLI main/subcommand, output formatters, integration tests (36 test cases).
**Files:** src/mlx_lm_lens/core/logit_projection.py, src/mlx_lm_lens/cli/{main,logit_lens,logit_lens_formatter}.py, tests/integration/test_logit_lens.py
**Decisions:** Numerically stable softmax, top-5 tokens per position, split formatters to keep files <200 lines, graceful fallback for chat templates

### Session 3 -- 2026-03-29: Phase 1d - Output formatter system
**What we did:** Built formatter abstraction with table/JSON/CSV implementations, registry pattern, 35 unit tests.
**Files:** formatters/{base,table,json,csv,registry}.py, tests/unit/formatters/test_formatters.py
**Decisions:** Protocol for duck typing, graceful empty input handling, DictWriter for CSV escaping, factory pattern for CLI integration

## Milestones

- [x] Phase 1: Refactor & Foundation (extract metrics, build abstractions, verify equivalence)
- [x] Phase 2: Logit Lens (implement CLI command + tests)
- [x] Phase 3: Activation Analysis (wire metrics, batch processing, anomaly detection)
- [x] Phase 4: Circuit Discovery (ablations, patching, decomposition, weight angles)
- [x] Phase 5: Polish & Publish (README, GitHub Actions, examples, final coverage check)

## Mistakes & Lessons

(None yet)

## Technical Debt & Future Ideas

- Consider lazy loading for large models (phase 2+)
- Profile memory usage on M1/M2/M3 Apple Silicon for large 70B+ models
- Add progress bar for long-running analyses (rich library supports this)
