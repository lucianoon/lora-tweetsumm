# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Overview

Parameter-efficient fine-tuning of T5-small for abstractive dialogue summarization on the TweetSumm customer-support dataset, using LoRA (via HuggingFace PEFT). The project is an end-to-end applied-ML repo: training, evaluation, a rank-ablation experiment, and two deployable demos. Optimized to run locally in ~1 minute on Apple Silicon (MPS).

Note: user-facing docs (README.md, notebook narrative comments) are in Portuguese; code, docstrings, and log messages are in English.

## Commands

All entrypoints run as modules from the project root (`python -m ...`), which is why `scripts/*.py` manipulate `sys.path` — the `E402` ruff rule is ignored for scripts to allow this.

```bash
# Install (editable). Add [dev] for lint/test, [demo] for Gradio + translation
pip install -e ".[dev]"

# Train (defaults: configs/default.yaml, 300 samples, 3 epochs)
python -m scripts.train
python -m scripts.train --config configs/fast.yaml
python -m scripts.train --merge          # fuse adapters into base weights after training

# Evaluate ROUGE against the latest checkpoint in training.output_dir
python -m scripts.evaluate
python -m scripts.evaluate --checkpoint checkpoints/t5-lora-tweetsumm/checkpoint-225
python -m scripts.evaluate --baseline    # also score untuned base T5 side-by-side

# Rank-ablation experiment (sweeps LoRA rank, writes JSON + PNG to results/)
python -m scripts.experiments --ranks 4 8 16 --fast

# Interactive Gradio demo
python -m scripts.demo --allow-untrained # run without a trained checkpoint

# Lint / format (must pass in CI — see .github/workflows/ci.yml)
ruff format --check src/ scripts/ tests/
ruff check src/ scripts/ tests/

# Tests
pytest -m "not slow"    # fast: pure logic + mocked datasets, ~2s (this is what CI runs)
pytest                  # full: also downloads real t5-small (test_model, test_inference are @slow)
pytest tests/test_config.py::test_name   # single test
```

## Architecture

The pipeline is a linear flow through `src/`, driven by a single typed config object. Each `src/` module is a pure library; each `scripts/` file is a thin CLI orchestrator that wires them together.

**Config is the spine.** `src/config.py` defines nested dataclasses (`Config` → `LoraParams`, `TrainingParams`, `InferenceParams`). `load_config(path)` reads YAML from `configs/` and pops the `lora`/`training`/`inference` sub-dicts into their dataclasses. Everything downstream takes a `Config`. Device is auto-detected in `Config.__post_init__` (MPS → CUDA → CPU) unless `device` is set explicitly.

**Data flow** (`src/data.py`): loads the HF dataset, auto-detects the dialogue column name (varies across dataset versions: `dialogue`/`conversation`/`dialog`), prepends the T5 task prefix (`"summarize: "`), tokenizes, and masks pad tokens in labels with `-100`. Train/eval are tokenized; **test is kept raw** (generation happens at eval time). Returns a dict including `dialogue_column`, which callers thread through to inference.

**Model** (`src/model.py`): `build_model` wraps base T5 with a PEFT `LoraConfig` (rsLoRA scaling, adapters on attention `q`/`v` projections). Key helpers: `find_latest_checkpoint` (picks the highest-numbered `checkpoint-*` dir containing `adapter_config.json`), `load_trained_model` (loads base + adapter for eval/inference; warns if adapter rank ≠ config rank and treats the adapter as source of truth), and `merge_and_save` (fuses adapters via `merge_and_unload` for zero-overhead inference).

**Training** (`src/train.py`): builds a `Seq2SeqTrainer` with `DataCollatorForSeq2Seq`, `predict_with_generate=True`, `save_strategy="epoch"`. `bf16` is enabled only on CUDA. `run_training` wraps `.train()` with wall-clock timing added to the metrics dict.

**Inference** (`src/inference.py`): `summarize` does prefix + tokenize + `model.generate` + decode. `compare_before_after` is used to show generation quality pre/post training in the train script.

### Conventions

- **Checkpoints are LoRA adapters, not full models.** A valid checkpoint dir must contain `adapter_config.json`. Generated checkpoints and `results/` artifacts are gitignored — regenerate by re-running training/experiments.
- Adding a config field means updating the relevant dataclass in `src/config.py` (and, if nested, the pop logic in `load_config`).
- Tests use the `sample_config` fixture (`tests/conftest.py`) — forces CPU, tiny sizes, no file I/O. Mark any test that downloads a model/dataset with `@pytest.mark.slow`.

## Deployment demos

Two separate demo builds (both gitignored from the main flow, self-contained dirs):
- `space-static/` — the live demo; runs entirely in-browser via Transformers.js (LoRA merged into T5-small, exported to ONNX, INT8-quantized). No server.
- `space/` — a server-side Gradio build (loads the adapter from HF Hub; hosting requires HF PRO — see `space/DEPLOY.md`).
