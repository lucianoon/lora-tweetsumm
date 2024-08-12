#!/usr/bin/env python3
"""
Train a T5 model with LoRA adapters on the TweetSumm dataset.

Usage:
    python -m scripts.train                          # uses configs/default.yaml
    python -m scripts.train --config configs/fast.yaml  # custom config

This script orchestrates the full pipeline:
    1. Load configuration from YAML
    2. Download and tokenize the dataset
    3. Build T5 + LoRA model
    4. Show pre-training generation (baseline)
    5. Run fine-tuning
    6. Show post-training generation (improved)
    7. Optionally merge adapters into base model
"""

from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path

# Ensure project root is on sys.path for clean imports
_PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(_PROJECT_ROOT))

from src.config import load_config
from src.data import load_data, load_tokenizer
from src.inference import compare_before_after, print_comparison
from src.model import build_model, merge_and_save
from src.train import create_trainer, run_training


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
        description="LoRA fine-tuning of T5 for dialogue summarization",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument(
        "--config",
        type=str,
        default=None,
        help="Path to YAML config file (default: configs/default.yaml)",
    )
    parser.add_argument(
        "--merge",
        action="store_true",
        help="Merge LoRA adapters into base model after training",
    )
    return parser.parse_args()


def main() -> None:
    """Main training entrypoint."""
    setup_logging()
    logger = logging.getLogger("train")

    args = parse_args()
    config = load_config(args.config)

    # ── Data ──────────────────────────────────────────────────
    tokenizer = load_tokenizer(config)
    data = load_data(config, tokenizer)

    # ── Model ─────────────────────────────────────────────────
    model = build_model(config)

    # ── Pre-training baseline ─────────────────────────────────
    test_sample = data["test"][0]
    dlg_col = data["dialogue_column"]

    before = compare_before_after(model, tokenizer, test_sample, dlg_col, config)
    print_comparison("BEFORE fine-tuning", before)

    # ── Training ──────────────────────────────────────────────
    trainer = create_trainer(
        model=model,
        tokenizer=tokenizer,
        train_dataset=data["train"],
        eval_dataset=data["eval"],
        config=config,
    )
    metrics = run_training(trainer)
    logger.info("Training metrics: %s", metrics)

    # ── Post-training result ──────────────────────────────────
    after = compare_before_after(model, tokenizer, test_sample, dlg_col, config)
    print_comparison("AFTER fine-tuning", after)

    # ── Merge (optional) ──────────────────────────────────────
    if args.merge:
        merge_path = Path(config.training.output_dir).parent / "t5-lora-merged"
        merge_and_save(model, merge_path)
        logger.info("Merged model saved to: %s", merge_path)

    logger.info("Done! ✓")


if __name__ == "__main__":
    main()
