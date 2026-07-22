"""
Model construction with LoRA adapters.

Builds a T5 Seq2Seq model, wraps it with PEFT LoRA adapters, and provides
utilities for merging adapters back into the base weights.
"""

from __future__ import annotations

import json
import logging
from pathlib import Path

from peft import LoraConfig, PeftModel, TaskType, get_peft_model
from transformers import AutoModelForSeq2SeqLM, PreTrainedModel

from src.config import Config

logger = logging.getLogger(__name__)


def build_model(config: Config) -> PreTrainedModel:
    """Build a T5 model with LoRA adapters applied.

    Loads the base model, configures LoRA with rank-stabilized scaling
    (rsLoRA), and moves the model to the target device.

    Args:
        config: Pipeline configuration with model and LoRA params.

    Returns:
        A PEFT-wrapped model ready for training.
    """
    logger.info("Loading base model: %s", config.model_id)
    base_model = AutoModelForSeq2SeqLM.from_pretrained(config.model_id)

    lora_config = LoraConfig(
        task_type=TaskType.SEQ_2_SEQ_LM,
        r=config.lora.r,
        lora_alpha=config.lora.alpha,
        lora_dropout=config.lora.dropout,
        target_modules=config.lora.target_modules,
        use_rslora=config.lora.use_rslora,
    )

    model = get_peft_model(base_model, lora_config)

    # Log trainable parameter count
    trainable, total = model.get_nb_trainable_parameters()
    pct = 100 * trainable / total
    logger.info(
        "LoRA applied — trainable: %s / %s (%.2f%%)",
        f"{trainable:,}",
        f"{total:,}",
        pct,
    )

    model.to(config.device)
    logger.info("Model moved to device: %s", config.device)

    return model


def find_latest_checkpoint(output_dir: str | Path) -> Path:
    """Find the newest HuggingFace Trainer checkpoint in an output directory.

    Args:
        output_dir: Directory containing ``checkpoint-*`` subdirectories.

    Returns:
        Path to the checkpoint with the largest numeric step.

    Raises:
        FileNotFoundError: If no adapter checkpoint is found.
    """
    root = Path(output_dir)
    checkpoints = []

    if root.exists():
        for path in root.glob("checkpoint-*"):
            if not path.is_dir() or not (path / "adapter_config.json").exists():
                continue
            try:
                step = int(path.name.rsplit("-", 1)[1])
            except (IndexError, ValueError):
                continue
            checkpoints.append((step, path))

    if not checkpoints:
        raise FileNotFoundError(
            f"No LoRA adapter checkpoints found in {root}. "
            "Run training first or pass --checkpoint with an adapter directory."
        )

    return max(checkpoints, key=lambda item: item[0])[1]


def read_adapter_rank(checkpoint: str | Path) -> int | None:
    """Read the LoRA rank stored in a PEFT adapter config.

    Args:
        checkpoint: Adapter directory containing ``adapter_config.json``.

    Returns:
        The configured LoRA rank, or ``None`` if it cannot be parsed.
    """
    adapter_config_path = Path(checkpoint) / "adapter_config.json"
    try:
        with open(adapter_config_path, encoding="utf-8") as f:
            adapter_config = json.load(f)
    except (FileNotFoundError, json.JSONDecodeError):
        return None

    rank = adapter_config.get("r")
    return rank if isinstance(rank, int) else None


def load_trained_model(config: Config, checkpoint: str | Path | None = None) -> PreTrainedModel:
    """Load a base T5 model with trained LoRA adapter weights.

    Args:
        config: Pipeline configuration.
        checkpoint: Adapter directory. If omitted, the latest checkpoint under
            ``config.training.output_dir`` is used.

    Returns:
        A PEFT model with trained adapter weights loaded for inference/eval.
    """
    checkpoint_path = (
        Path(checkpoint) if checkpoint else find_latest_checkpoint(config.training.output_dir)
    )
    if not (checkpoint_path / "adapter_config.json").exists():
        raise FileNotFoundError(
            f"{checkpoint_path} does not look like a LoRA adapter directory "
            "(missing adapter_config.json)."
        )

    adapter_rank = read_adapter_rank(checkpoint_path)
    if adapter_rank is not None and adapter_rank != config.lora.r:
        logger.warning(
            "Adapter rank (%d) differs from config rank (%d). "
            "Using the adapter checkpoint as the source of truth.",
            adapter_rank,
            config.lora.r,
        )

    logger.info("Loading base model: %s", config.model_id)
    base_model = AutoModelForSeq2SeqLM.from_pretrained(config.model_id)
    logger.info("Loading trained LoRA adapter from: %s", checkpoint_path)
    model = PeftModel.from_pretrained(base_model, checkpoint_path)
    model.to(config.device)
    model.eval()
    logger.info("Trained model moved to device: %s", config.device)
    return model


def get_model_stats(model: PreTrainedModel) -> dict:
    """Collect model parameter statistics.

    Args:
        model: A PEFT-wrapped model.

    Returns:
        A dict with ``total_params``, ``trainable_params``,
        ``trainable_pct``, and ``model_size_mb``.
    """
    trainable, total = model.get_nb_trainable_parameters()
    if trainable == 0 and hasattr(model, "peft_config"):
        trainable = sum(p.numel() for name, p in model.named_parameters() if "lora_" in name)
    size_mb = sum(p.numel() * p.element_size() for p in model.parameters()) / (1024**2)

    return {
        "total_params": total,
        "trainable_params": trainable,
        "trainable_pct": round(100 * trainable / total, 4),
        "model_size_mb": round(size_mb, 2),
    }


def merge_and_save(model: PreTrainedModel, output_dir: str | Path) -> Path:
    """Merge LoRA adapters into the base model and save.

    Fusing the adapters removes any inference overhead — the resulting
    model is identical in architecture to the original T5 but with
    updated weights.

    Args:
        model: A PEFT-wrapped model after training.
        output_dir: Directory to save the merged model.

    Returns:
        Path to the saved model directory.
    """
    output_path = Path(output_dir)
    logger.info("Merging LoRA adapters and saving to: %s", output_path)

    merged = model.merge_and_unload()
    merged.save_pretrained(output_path)

    logger.info("Merged model saved successfully.")
    return output_path
