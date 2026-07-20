"""
Training pipeline using HuggingFace Seq2SeqTrainer.

Configures the trainer with the correct arguments, data collator,
and evaluation strategy based on the YAML config.
"""

from __future__ import annotations

import logging
import time

from transformers import (
    DataCollatorForSeq2Seq,
    PreTrainedModel,
    PreTrainedTokenizerBase,
    Seq2SeqTrainer,
    Seq2SeqTrainingArguments,
)

from src.config import Config

logger = logging.getLogger(__name__)


def create_trainer(
    model: PreTrainedModel,
    tokenizer: PreTrainedTokenizerBase,
    train_dataset,
    eval_dataset,
    config: Config,
) -> Seq2SeqTrainer:
    """Create a fully-configured Seq2SeqTrainer.

    Args:
        model: The PEFT-wrapped model to train.
        tokenizer: Tokenizer for the data collator.
        train_dataset: Tokenized training dataset.
        eval_dataset: Tokenized evaluation dataset.
        config: Pipeline configuration.

    Returns:
        A ready-to-train :class:`Seq2SeqTrainer`.
    """
    training_args = Seq2SeqTrainingArguments(
        output_dir=config.training.output_dir,
        num_train_epochs=config.training.epochs,
        per_device_train_batch_size=config.training.batch_size,
        learning_rate=config.training.learning_rate,
        predict_with_generate=True,
        logging_steps=config.training.logging_steps,
        eval_strategy=config.training.eval_strategy,
        report_to="none",
        bf16=(config.device == "cuda" and config.training.bf16_on_cuda),
        save_strategy="epoch",
        save_total_limit=2,
        load_best_model_at_end=False,
        remove_unused_columns=True,
    )

    data_collator = DataCollatorForSeq2Seq(
        tokenizer,
        model=model,
        pad_to_multiple_of=8,
    )

    trainer = Seq2SeqTrainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=eval_dataset,
        data_collator=data_collator,
    )

    logger.info(
        "Trainer created — epochs=%d, batch_size=%d, lr=%.1e",
        config.training.epochs,
        config.training.batch_size,
        config.training.learning_rate,
    )

    return trainer


def run_training(trainer: Seq2SeqTrainer) -> dict:
    """Execute the training loop.

    Args:
        trainer: A configured Seq2SeqTrainer.

    Returns:
        Training metrics dictionary enriched with wall-clock timing:
        ``wall_time_seconds`` and ``samples_per_second``.
    """
    logger.info("Starting training...")

    t0 = time.perf_counter()
    result = trainer.train()
    wall_time = time.perf_counter() - t0

    metrics = result.metrics
    metrics["wall_time_seconds"] = round(wall_time, 2)

    logger.info(
        "Training complete — loss=%.4f, wall_time=%.1fs, runtime=%.1fs",
        metrics.get("train_loss", float("nan")),
        wall_time,
        metrics.get("train_runtime", 0),
    )

    return metrics
