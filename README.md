# mlx-lm-lens

Mechanistic interpretability CLI for transformer models on Apple Silicon. Analyze per-layer predictions, monitor activation drift against reference models, and discover model circuits.

**Author:** [Adeel Ahmad](mailto:adeelahmad99@gmail.com)

## Features

- **Logit Lens** — Stream token generation with per-layer top-K predictions at each position
- **Drift Correction** — Measure and correct activation angular drift against reference models (base, adapters, quantized variants)
- **Rich TUI** — Live terminal UI with colored streaming, scrollable output, markdown support, real-time drift metrics
- **Reference Model Comparison** — Compare fine-tuned/adapted models against base models in activation space
- **LoRA & Quantization Support** — Transparent handling of 4-bit, 8-bit, and LoRA adapters via mlx-lm
- **Circuit Discovery** — Ablate layers to trace information flow
- **JSON Logging** — Export per-layer angles, predictions, and timing for analysis

## Why It Matters

Modern language models are black boxes. mlx-lm-lens provides:

- **Mechanistic interpretability on consumer hardware** — M1/M2/M3 chips via MLX, no GPU needed
- **Understand fine-tuning effects** — See how activation space changes after fine-tuning or adapter application
- **Debug model behavior** — Track prediction confidence changes across layers, spot layer collapse
- **Reference-based drift detection** — Quantify how adapted models diverge from base models, apply geometric corrections

## Installation

**Requirements:** Python ≥3.10, Apple Silicon (M1/M2/M3)

```bash
git clone https://github.com/adeelahmad/mlx-lm-lens.git
cd mlx-lm-lens
pip install -e .
```

Or from GitHub directly:
```bash
pip install git+https://github.com/adeelahmad/mlx-lm-lens.git
```

## Quick Start

### Generate with logit lens

```bash
mlx-lm-lens logit-lens generate \
  --model /path/to/model \
  --prompt "The future of AI is" \
  --max-tokens 50 \
  --top-k 5
```

Output: Live TUI with colored generation stream, per-layer predictions, agreement metrics.

### Compare models with drift correction

```bash
mlx-lm-lens logit-lens generate \
  --model /path/to/finetuned-model \
  --reference-model /path/to/base-model \
  --drift-correction \
  --drift-threshold 0.3 \
  --drift-log /tmp/drift.jsonl \
  --prompt "Explain quantum mechanics" \
  --max-tokens 100 \
  --no-tui
```

Measures per-layer activation angles against reference baselines. Logs angles to JSONL for analysis.

### Analyze activations

```bash
mlx-lm-lens activations \
  --model /path/to/model \
  --prompt "Your prompt" \
  --metrics cosine,cka,mad
```
```
Layer  Top-1 Token  Confidence  Flip?  Top-2           Top-3
─────  ────────────  ──────────  ─────  ──────────────  ──────────────
0      " the"         0.23       No     " a"            " is"
1      " answer"      0.31       No     " that"         " is"
...
15     "4"            0.87       Yes    " four"         " 4"
```

### Activation Analysis: Compare Model States

Measure how similar hidden states are between two models across layers.

```bash
mlx-lm-lens activations \
  --model /path/to/model1 \
  --reference-model /path/to/model2 \
  --prompt "Hello world" \
  --metrics cosine,cka,procrustes
```

Output:
```
Layer  Cosine Sim  CKA     Procrustes  MAD
─────  ──────────  ──────  ──────────  ──────
0      0.94        0.91    0.12        0.08
1      0.89        0.84    0.18        0.11
...
15     0.42        0.31    0.58        0.45
```

### Circuit Discovery: Find Important Layers

Ablate each layer and measure its impact on the final prediction.

```bash
mlx-lm-lens circuit ablate \
  --model /path/to/model \
  --prompt "The capital of France is" \
  --method zero
```

Output:
```
Layer  Importance (KL)  Rank  Impact
─────  ────────────────  ────  ──────────────────────
15     2.34              1     High
14     1.89              2     High
13     0.92              3     Medium
...
```

## Complete Feature Reference

### Logit Lens

```bash
mlx-lm-lens logit-lens \
  --model MODEL_PATH \                    # Required: path to model directory
  --prompt PROMPT \                       # Required: input text
  --adapter ADAPTER_PATH \                # Optional: path to LoRA adapter
  --top-k 5 \                             # How many top predictions to show (default: 5)
  --all-positions \                       # Show predictions at every position (default: last only)
  --compare-base \                        # Compare predictions with base model (requires --adapter)
  --use-chat-template \                   # Apply chat template before tokenization
  --use-tuned-lens \                      # Use tuned lens projection (if available)
  --format table|json|csv \               # Output format (default: table)
  --output FILE \                         # Write to file instead of stdout
  --verbose                               # Enable debug logging
```

### Activation Analysis

```bash
mlx-lm-lens activations \
  --model MODEL_PATH \                    # Required: primary model
  --reference-model REF_PATH \            # Optional: second model for comparison
  --prompt PROMPT \                       # Required: input text
  --metrics METRIC_LIST \                 # Required: comma-separated metric names
  --batch-prompts FILE \                  # Optional: JSONL file with multiple prompts
  --adapter ADAPTER_PATH \                # Optional: LoRA adapter for primary model
  --format table|json|csv \               # Output format (default: table)
  --output DIR \                          # Write results to directory
  --verbose
```

**Available metrics:**
- `cosine` — Cosine similarity in activation space
- `cka` — Centered Kernel Alignment (RBF kernel)
- `procrustes` — Procrustes distance (alignment cost)
- `grassmannian` — Principal angles between subspaces
- `mad` — Mean Absolute Deviation
- `effective-dim` — Effective dimensionality (fraction of spectrum)
- `energy-kl` — Energy-based KL divergence
- `rsa` — Representational Similarity Analysis

### Circuit Discovery — Ablate

```bash
mlx-lm-lens circuit ablate \
  --model MODEL_PATH \
  --prompt PROMPT \
  --method zero|mean|noise|knockout \     # Ablation strategy (default: zero)
  --target-token TOKEN \                  # Optional: specific token to measure
  --layers START-END \                    # Optional: layer range (e.g., "10-20")
  --format table|json|csv \
  --output DIR
```

**Ablation methods:**
- `zero` — Replace layer output with zeros
- `mean` — Replace with mean activation across positions
- `noise` — Replace with Gaussian noise
- `knockout` — Zero out residual stream at layer input

### Circuit Discovery — Other Subcommands

```bash
mlx-lm-lens circuit patch \
  --model MODEL_PATH \
  --reference REF_PATH \
  --prompt PROMPT \
  --patch-layer N \                       # Optional: patch specific layer only
  --sweep                                 # Sweep through all layers

mlx-lm-lens circuit decompose \
  --model MODEL_PATH \
  --prompt PROMPT \
  --position N \                          # Token position (-1 = last)
  --show-contributions

mlx-lm-lens circuit angles \
  --base-adapter PATH \
  --current-adapter PATH \
  --per-layer
```

## Examples

### Use Case 1: Understanding Qwen 0.6B Predictions

**Scenario:** You want to see which layer in Qwen3-0.6B decides to output "Paris" when asked "The capital of France is..."

```bash
mlx-lm-lens logit-lens \
  --model /path/to/qwen3-0.6b \
  --prompt "The capital of France is" \
  --top-k 3
```

**What to expect:** Early layers (0-5) might predict generic tokens like "the" or "a". Middle layers (6-12) start hinting at geography. By layer 15, "Paris" dominates with >90% confidence.

**Insight:** If "Paris" appears early and stays high, the model is confident and hierarchical. If it flips multiple times, the model is uncertain and corrects itself layer-to-layer.

### Use Case 2: Comparing Base vs Fine-Tuned Models

**Scenario:** You fine-tuned a model on math problems. Where do the activations diverge most?

```bash
# First, compute activation analysis on the fine-tuned version
mlx-lm-lens activations \
  --model /path/to/finetuned \
  --reference-model /path/to/base \
  --prompt "What is the derivative of x^2?" \
  --metrics cosine,cka,procrustes \
  --output results/

# Look at the CSV to find layers with largest divergence
cat results/activations.csv | sort -t',' -k3 -n
```

**What to expect:** Early layers (embeddings, first 5 layers) show high similarity (cosine sim >0.9). Middle layers diverge (0.7-0.8). Head layers might collapse to near-zero similarity if fine-tuning specialized them.

**Insight:** Large drops in CKA/Procrustes indicate layers that the fine-tuning process restructured most. These are likely domain-specific feature detectors.

### Use Case 3: Finding Identity Layers via Ablation

**Scenario:** Your model sometimes fails on identity tasks (copying the input). Which layers are responsible?

```bash
mlx-lm-lens circuit ablate \
  --model /path/to/model \
  --prompt "repeat: hello -> hello" \
  --method zero \
  --output results/

# Interpret: high importance = ablating it hurts identity task
```

**What to expect:** Layers 8-12 likely show high importance (KL divergence >1.0 when ablated). Removing these layers breaks the identity behavior.

**Insight:** These layers contain the "copy" circuit. Visualize their attention patterns to understand the mechanism.

## Architecture

mlx-lm-lens is organized into modular, testable components:

- **`core/`** — Model loading, activation capture, logit projection (low-level APIs)
- **`metrics/`** — 9 pluggable similarity metrics with unified interface
- **`ablations/`** — 4 ablation strategies for circuit discovery
- **`formatters/`** — Output to table, JSON, CSV
- **`cli/`** — User-facing commands (typer-based)
- **`shaman/`** — Hypothesis validators (H1-H15 from SHAMAN framework)
- **`circuit/`** — Patching and residual stream decomposition

Data flows: Model → Activation Capture → Metrics/Ablations → Formatters → User.

All components are dependency-injected via config dataclasses (no global state).

## Troubleshooting

### "CUDA not available" or MLX import fails
**Cause:** MLX only runs on Apple Silicon (M1/M2/M3). Linux/Windows users need Docker with Rosetta emulation or similar.
**Fix:** Ensure you're on a Mac with Apple Silicon. Check: `python -c "import mlx.core; print(mlx.core.default_device())"`

### "scipy module not found" — RSA metric fails
**Cause:** RSA requires scipy but it's optional.
**Fix:** Install dev dependencies: `pip install -e .[dev]` or manually: `pip install scipy`

### Memory errors on large models
**Cause:** Quantized models (4-bit/8-bit) still use memory for activations. Typical 70B model needs 16+ GB.
**Fix:**
- Use smaller models for exploration (0.6B-7B)
- Use `--all-positions` sparingly (captures all token positions, not just last)
- Run on a Mac with more memory (M3 Max with 96GB)

### "Top prediction never flips" in logit lens
**Cause:** Either the model is very confident, or the prompt is ambiguous.
**Fix:** Try prompts with more open-ended predictions (e.g., story continuations instead of facts).

### "Reference model not loaded" error
**Cause:** Activation analysis requires both models to be the same architecture (same layer count, hidden size).
**Fix:** Ensure both model paths are correct and compatible. Use `mlx-lm-lens activations --model M1 --reference-model M2 --prompt "x"` to verify both load.

## Contributing

We welcome contributions! See [CONTRIBUTING.md](CONTRIBUTING.md) for development setup, testing, and style guidelines.

Key principles:
- Every change must maintain ≥90% test coverage
- New features require tests and documentation
- All code passes strict ruff linter (zero warnings)
- Keep files under 200 lines (forces modularity)

## License

Apache License 2.0. See [LICENSE](LICENSE) for details.

---

**Built for researchers and practitioners** who want to understand transformer internals without a GPU farm.

Questions? Open an issue or check the [examples/](examples/) directory for more use cases.
