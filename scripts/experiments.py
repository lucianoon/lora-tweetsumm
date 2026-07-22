#!/usr/bin/env python3
"""
Rank ablation experiment for LoRA fine-tuning.

Trains and evaluates the model across multiple LoRA ranks to find
the optimal trade-off between parameter efficiency and quality.

Usage:
    python -m scripts.experiments                       # ranks 4,8,16,32
    python -m scripts.experiments --ranks 4 8 16        # custom ranks
    python -m scripts.experiments --fast                 # use fast.yaml config
    python -m scripts.experiments --ranks 4 8 --fast     # combine
"""

from __future__ import annotations

import argparse
import json
import logging
import sys
from datetime import datetime, timezone
from pathlib import Path

_PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(_PROJECT_ROOT))

import evaluate  # noqa: I001
import matplotlib

matplotlib.use("Agg")  # Non-interactive backend for server/CI environments

import matplotlib.pyplot as plt  # noqa: E402

from src.config import load_config
from src.data import load_data, load_tokenizer
from src.inference import summarize
from src.model import build_model, get_model_stats
from src.train import create_trainer, run_training

logger = logging.getLogger(__name__)

DEFAULT_RANKS = [4, 8, 16, 32]


def setup_logging() -> None:
    """Configure structured logging to stdout."""
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s │ %(name)-20s │ %(levelname)-5s │ %(message)s",
        datefmt="%H:%M:%S",
        handlers=[logging.StreamHandler(sys.stdout)],
    )


def run_single_experiment(rank: int, config_path: str | None) -> dict:
    """Run a full train+evaluate cycle for a specific LoRA rank.

    Args:
        rank: The LoRA rank to use.
        config_path: Path to the base YAML config file.

    Returns:
        Dict with rank, model stats, training metrics, and ROUGE scores.
    """
    logger.info("=" * 60)
    logger.info("  EXPERIMENT: LoRA rank = %d", rank)
    logger.info("=" * 60)

    # Load config and override rank
    config = load_config(config_path)
    config.lora.r = rank
    config.training.output_dir = str(Path(config.training.output_dir) / f"rank-{rank}")

    # ── Data ──────────────────────────────────────────────────
    tokenizer = load_tokenizer(config)
    data = load_data(config, tokenizer)

    # ── Model ─────────────────────────────────────────────────
    model = build_model(config)
    stats = get_model_stats(model)
    logger.info(
        "Model stats — trainable: %s (%.2f%%)",
        f"{stats['trainable_params']:,}",
        stats["trainable_pct"],
    )

    # ── Train ─────────────────────────────────────────────────
    trainer = create_trainer(
        model=model,
        tokenizer=tokenizer,
        train_dataset=data["train"],
        eval_dataset=data["eval"],
        config=config,
    )
    train_metrics = run_training(trainer)

    # ── Evaluate (ROUGE) ──────────────────────────────────────
    test_ds = data["test"]
    dlg_col = data["dialogue_column"]

    predictions = []
    references = []
    for sample in test_ds:
        pred = summarize(model, tokenizer, sample[dlg_col], config)
        predictions.append(pred)
        references.append(sample["summary"])

    rouge = evaluate.load("rouge")
    scores = rouge.compute(predictions=predictions, references=references)

    result = {
        "rank": rank,
        "alpha": config.lora.alpha,
        "model_stats": stats,
        "train_loss": round(train_metrics.get("train_loss", 0), 4),
        "wall_time_seconds": train_metrics.get("wall_time_seconds", 0),
        "rouge1": round(scores["rouge1"], 4),
        "rouge2": round(scores["rouge2"], 4),
        "rougeL": round(scores["rougeL"], 4),
        "rougeLsum": round(scores["rougeLsum"], 4),
    }

    logger.info(
        "Rank %d results — ROUGE-L=%.4f, loss=%.4f, time=%.1fs",
        rank,
        result["rougeL"],
        result["train_loss"],
        result["wall_time_seconds"],
    )

    return result


def plot_results(results: list[dict], output_path: Path) -> None:
    """Generate a publication-quality ablation plot.

    Creates a dual-axis chart: ROUGE scores vs rank (left axis)
    and trainable parameters vs rank (right axis).

    Args:
        results: List of experiment result dicts.
        output_path: Path to save the PNG plot.
    """
    ranks = [r["rank"] for r in results]
    rouge1 = [r["rouge1"] for r in results]
    rouge2 = [r["rouge2"] for r in results]
    rougeL = [r["rougeL"] for r in results]
    trainable = [r["model_stats"]["trainable_params"] for r in results]

    fig, ax1 = plt.subplots(figsize=(10, 6))

    # ── Style ─────────────────────────────────────────────────
    fig.patch.set_facecolor("#0f172a")
    ax1.set_facecolor("#1e293b")

    colors = {
        "rouge1": "#818cf8",
        "rouge2": "#34d399",
        "rougeL": "#f472b6",
        "params": "#fbbf24",
    }

    # ── ROUGE lines (left axis) ───────────────────────────────
    ax1.plot(
        ranks,
        rouge1,
        "o-",
        color=colors["rouge1"],
        linewidth=2.5,
        markersize=8,
        label="ROUGE-1",
        zorder=5,
    )
    ax1.plot(
        ranks,
        rouge2,
        "s-",
        color=colors["rouge2"],
        linewidth=2.5,
        markersize=8,
        label="ROUGE-2",
        zorder=5,
    )
    ax1.plot(
        ranks,
        rougeL,
        "D-",
        color=colors["rougeL"],
        linewidth=2.5,
        markersize=8,
        label="ROUGE-L",
        zorder=5,
    )

    ax1.set_xlabel("LoRA Rank (r)", fontsize=13, color="#e2e8f0", fontweight="bold")
    ax1.set_ylabel("ROUGE Score", fontsize=13, color="#e2e8f0", fontweight="bold")
    ax1.set_xticks(ranks)
    ax1.tick_params(colors="#94a3b8")
    ax1.grid(True, alpha=0.15, color="#475569")

    # ── Trainable params (right axis) ─────────────────────────
    ax2 = ax1.twinx()
    ax2.bar(
        ranks,
        trainable,
        alpha=0.2,
        color=colors["params"],
        width=[r * 0.3 for r in ranks],
        label="Trainable Params",
        zorder=1,
    )
    ax2.set_ylabel("Trainable Parameters", fontsize=13, color="#e2e8f0", fontweight="bold")
    ax2.tick_params(colors="#94a3b8")

    # Format y-axis with K notation
    ax2.yaxis.set_major_formatter(plt.FuncFormatter(lambda x, _: f"{x / 1000:.0f}K"))

    # ── Legend ────────────────────────────────────────────────
    lines1, labels1 = ax1.get_legend_handles_labels()
    lines2, labels2 = ax2.get_legend_handles_labels()
    ax1.legend(
        lines1 + lines2,
        labels1 + labels2,
        loc="upper left",
        framealpha=0.8,
        facecolor="#1e293b",
        edgecolor="#334155",
        labelcolor="#e2e8f0",
        fontsize=11,
    )

    # ── Title ─────────────────────────────────────────────────
    ax1.set_title(
        "LoRA Rank Ablation — ROUGE vs Trainable Parameters",
        fontsize=16,
        color="#f8fafc",
        fontweight="bold",
        pad=20,
    )

    plt.tight_layout()
    fig.savefig(output_path, dpi=150, bbox_inches="tight", facecolor=fig.get_facecolor())
    plt.close(fig)

    logger.info("Ablation plot saved to: %s", output_path)


def print_results_table(results: list[dict]) -> None:
    """Print a formatted results table to stdout."""
    print("\n" + "═" * 80)
    print("  LoRA Rank Ablation — Results Summary")
    print("═" * 80)
    header = (
        f"  {'Rank':>4}  │ {'Params':>10}  │ {'%':>6}  "
        f"│ {'ROUGE-1':>8}  │ {'ROUGE-2':>8}  │ {'ROUGE-L':>8}  │ {'Time(s)':>8}"
    )
    print(header)
    print("─" * 80)

    for r in results:
        print(
            f"  r={r['rank']:<3d} │ "
            f"{r['model_stats']['trainable_params']:>10,}  │ "
            f"{r['model_stats']['trainable_pct']:>5.2f}%  │ "
            f"{r['rouge1']:>8.4f}  │ "
            f"{r['rouge2']:>8.4f}  │ "
            f"{r['rougeL']:>8.4f}  │ "
            f"{r['wall_time_seconds']:>7.1f}s"
        )

    print("═" * 80)

    # Find best rank by ROUGE-L
    best = max(results, key=lambda x: x["rougeL"])
    print(f"\n  🏆 Best rank by ROUGE-L: r={best['rank']} (ROUGE-L={best['rougeL']:.4f})")
    print()


def parse_args() -> argparse.Namespace:
    """Parse command-line arguments."""
    parser = argparse.ArgumentParser(
        description="LoRA rank ablation experiment",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument(
        "--ranks",
        type=int,
        nargs="+",
        default=DEFAULT_RANKS,
        help=f"LoRA ranks to test (default: {DEFAULT_RANKS})",
    )
    parser.add_argument(
        "--config",
        type=str,
        default=None,
        help="Path to YAML config file (default: configs/default.yaml)",
    )
    parser.add_argument(
        "--fast",
        action="store_true",
        help="Use configs/fast.yaml for quick iteration",
    )
    return parser.parse_args()


def main() -> None:
    """Run the rank ablation experiment."""
    setup_logging()

    args = parse_args()

    config_path = args.config
    if args.fast and config_path is None:
        config_path = str(_PROJECT_ROOT / "configs" / "fast.yaml")
        logger.info("Using fast config: %s", config_path)

    # ── Run experiments ───────────────────────────────────────
    results = []
    for rank in sorted(args.ranks):
        result = run_single_experiment(rank, config_path)
        results.append(result)

    # ── Display ───────────────────────────────────────────────
    print_results_table(results)

    # ── Save results ──────────────────────────────────────────
    config = load_config(config_path)
    results_dir = Path(config.results_dir)
    results_dir.mkdir(parents=True, exist_ok=True)

    timestamp = datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")

    # JSON results
    results_path = results_dir / f"rank_ablation_{timestamp}.json"
    payload = {
        "experiment": "rank_ablation",
        "timestamp": timestamp,
        "model_id": config.model_id,
        "n_train": config.n_train,
        "epochs": config.training.epochs,
        "ranks_tested": sorted(args.ranks),
        "results": results,
    }
    with open(results_path, "w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2)
    logger.info("Results saved to: %s", results_path)

    # Plot
    plot_path = results_dir / "rank_ablation.png"
    plot_results(results, plot_path)

    # Also save a "latest" symlink-style copy for README embedding
    latest_json = results_dir / "rank_ablation_latest.json"
    with open(latest_json, "w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2)

    logger.info("Experiment complete! ✓")


if __name__ == "__main__":
    main()
