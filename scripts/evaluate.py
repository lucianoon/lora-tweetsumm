#!/usr/bin/env python3
"""
Evaluate a fine-tuned T5 + LoRA model on the TweetSumm test set.

Computes ROUGE-1, ROUGE-2, and ROUGE-L scores with optional baseline
comparison against the untuned T5 model. Saves detailed per-sample
results to the configured results directory.

Usage:
    python -m scripts.evaluate
    python -m scripts.evaluate --config configs/default.yaml
    python -m scripts.evaluate --baseline   # also evaluate base T5
"""

from __future__ import annotations

import argparse
import json
import logging
import sys
from datetime import datetime, timezone
from pathlib import Path

# Ensure project root is on sys.path
_PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(_PROJECT_ROOT))

import evaluate
from tqdm import tqdm

from src.config import load_config
from src.data import load_data, load_tokenizer
from src.inference import summarize
from src.model import get_model_stats, load_trained_model


def setup_logging() -> None:
    """Configure structured logging to stdout."""
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s │ %(name)-20s │ %(levelname)-5s │ %(message)s",
        datefmt="%H:%M:%S",
        handlers=[logging.StreamHandler(sys.stdout)],
    )


def parse_args() -> argparse.Namespace:
    """Parse command-line arguments."""
    parser = argparse.ArgumentParser(
        description="Evaluate fine-tuned T5 + LoRA on TweetSumm test set",
    )
    parser.add_argument(
        "--config",
        type=str,
        default=None,
        help="Path to YAML config file (default: configs/default.yaml)",
    )
    parser.add_argument(
        "--baseline",
        action="store_true",
        help="Also evaluate the base T5 model (no LoRA) for comparison",
    )
    parser.add_argument(
        "--checkpoint",
        type=str,
        default=None,
        help=(
            "LoRA adapter checkpoint directory. Defaults to the latest checkpoint under "
            "training.output_dir."
        ),
    )
    return parser.parse_args()


def generate_predictions(
    model,
    tokenizer,
    test_ds,
    dlg_col: str,
    config,
    label: str,
) -> list[str]:
    """Generate summaries for all test samples with a progress bar.

    Args:
        model: Model to use for generation.
        tokenizer: Corresponding tokenizer.
        test_ds: Test dataset with raw dialogues.
        dlg_col: Name of the dialogue column.
        config: Pipeline configuration.
        label: Label for the progress bar.

    Returns:
        List of generated summary strings.
    """
    predictions = []
    for sample in tqdm(test_ds, desc=f"  {label}", unit="sample"):
        pred = summarize(model, tokenizer, sample[dlg_col], config)
        predictions.append(pred)
    return predictions


def compute_rouge(predictions: list[str], references: list[str]) -> dict:
    """Compute ROUGE scores.

    Args:
        predictions: Generated summaries.
        references: Reference summaries.

    Returns:
        Dict with ROUGE-1, ROUGE-2, ROUGE-L, ROUGE-Lsum scores.
    """
    rouge = evaluate.load("rouge")
    scores = rouge.compute(predictions=predictions, references=references)
    return {k: round(v, 4) for k, v in scores.items()}


def print_scores(label: str, scores: dict) -> None:
    """Pretty-print ROUGE scores for a model."""
    print(f"\n{'═' * 50}")
    print(f"  {label}")
    print("═" * 50)
    for metric, value in sorted(scores.items()):
        print(f"  {metric:12s}  {value:.4f}")
    print("═" * 50)


def print_comparison(fine_tuned_scores: dict, baseline_scores: dict) -> None:
    """Print a side-by-side comparison table."""
    print(f"\n{'═' * 60}")
    print("  Comparison: Base T5 vs Fine-Tuned (LoRA)")
    print("═" * 60)
    print(f"  {'Metric':12s}  │ {'Base T5':>10s}  │ {'Fine-Tuned':>10s}  │ {'Δ':>8s}")
    print("─" * 60)

    for metric in sorted(fine_tuned_scores.keys()):
        base_val = baseline_scores.get(metric, 0)
        ft_val = fine_tuned_scores[metric]
        delta = ft_val - base_val
        sign = "+" if delta >= 0 else ""
        print(f"  {metric:12s}  │ {base_val:>10.4f}  │ {ft_val:>10.4f}  │ {sign}{delta:>7.4f}")

    print("═" * 60)


def main() -> None:
    """Main evaluation entrypoint."""
    setup_logging()
    logger = logging.getLogger("evaluate")

    args = parse_args()
    config = load_config(args.config)

    # ── Load components ───────────────────────────────────────
    tokenizer = load_tokenizer(config)
    data = load_data(config, tokenizer)
    model = load_trained_model(config, args.checkpoint)
    model_stats = get_model_stats(model)

    test_ds = data["test"]
    dlg_col = data["dialogue_column"]
    references = [sample["summary"] for sample in test_ds]

    # ── Generate fine-tuned predictions ───────────────────────
    logger.info("Generating summaries for %d test samples...", len(test_ds))
    ft_predictions = generate_predictions(model, tokenizer, test_ds, dlg_col, config, "Fine-Tuned")
    ft_scores = compute_rouge(ft_predictions, references)
    print_scores("ROUGE — Fine-Tuned Model (LoRA)", ft_scores)

    # ── Baseline evaluation (optional) ────────────────────────
    baseline_scores = None
    baseline_predictions = None
    if args.baseline:
        from transformers import AutoModelForSeq2SeqLM

        logger.info("Loading base model for baseline comparison...")
        base_model = AutoModelForSeq2SeqLM.from_pretrained(config.model_id)
        base_model.to(config.device)

        baseline_predictions = generate_predictions(
            base_model, tokenizer, test_ds, dlg_col, config, "Base T5"
        )
        baseline_scores = compute_rouge(baseline_predictions, references)
        print_scores("ROUGE — Base T5 (no fine-tuning)", baseline_scores)

        # Side-by-side comparison
        print_comparison(ft_scores, baseline_scores)

    # ── Save detailed results ─────────────────────────────────
    results_dir = Path(config.results_dir)
    results_dir.mkdir(parents=True, exist_ok=True)

    timestamp = datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")

    # Per-sample details
    samples_detail = []
    for i, sample in enumerate(test_ds):
        dlg_text = sample[dlg_col]
        dlg_preview = dlg_text[:200] + "..." if len(dlg_text) > 200 else dlg_text
        entry = {
            "index": i,
            "dialogue": dlg_preview,
            "reference": references[i],
            "fine_tuned_prediction": ft_predictions[i],
        }
        if baseline_predictions:
            entry["baseline_prediction"] = baseline_predictions[i]
        samples_detail.append(entry)

    results_path = results_dir / f"evaluation_{timestamp}.json"
    results_payload = {
        "model_id": config.model_id,
        "n_test": len(test_ds),
        "timestamp": timestamp,
        "model_stats": model_stats,
        "fine_tuned_scores": ft_scores,
        "baseline_scores": baseline_scores,
        "config": {
            "lora_r": config.lora.r,
            "lora_alpha": config.lora.alpha,
            "epochs": config.training.epochs,
            "learning_rate": config.training.learning_rate,
            "n_train": config.n_train,
        },
        "samples": samples_detail,
    }

    with open(results_path, "w") as f:
        json.dump(results_payload, f, indent=2, ensure_ascii=False)

    logger.info("Detailed results saved to: %s", results_path)
    logger.info("Evaluation complete! ✓")


if __name__ == "__main__":
    main()
