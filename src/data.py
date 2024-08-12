"""
Data loading and preprocessing for TweetSumm dialogue summarization.

Handles dataset download, column name detection, tokenization, and
train/eval/test split preparation.
"""

from __future__ import annotations

import logging
from typing import Any, Dict

from datasets import DatasetDict, load_dataset
from transformers import AutoTokenizer, PreTrainedTokenizerBase

from src.config import Config

logger = logging.getLogger(__name__)


def _detect_dialogue_column(dataset: Any) -> str:
    """Detect the dialogue column name (varies across dataset versions)."""
    columns = dataset["train"].column_names
    for candidate in ("dialogue", "conversation", "dialog"):
        if candidate in columns:
            return candidate
    raise ValueError(f"Could not find a dialogue column. Available columns: {columns}")


def load_tokenizer(config: Config) -> PreTrainedTokenizerBase:
    """Load the tokenizer for the configured model.

    Args:
        config: Pipeline configuration.

    Returns:
        A HuggingFace tokenizer instance.
    """
    tokenizer = AutoTokenizer.from_pretrained(config.model_id)
    logger.info("Tokenizer loaded: %s (vocab_size=%d)", config.model_id, tokenizer.vocab_size)
    return tokenizer


def load_data(
    config: Config,
    tokenizer: PreTrainedTokenizerBase,
) -> Dict[str, Any]:
    """Load and tokenize the TweetSumm dataset.

    Args:
        config: Pipeline configuration with dataset params.
        tokenizer: Tokenizer to use for encoding.

    Returns:
        A dict with keys ``"train"``, ``"eval"``, ``"test"``, and
        ``"dialogue_column"`` (the detected column name).
    """
    logger.info("Loading dataset: %s", config.dataset_name)
    ds: DatasetDict = load_dataset(config.dataset_name)

    dlg_col = _detect_dialogue_column(ds)
    logger.info("Dialogue column detected: '%s'", dlg_col)

    def tokenize(batch: Dict[str, Any]) -> Dict[str, Any]:
        """Tokenize a batch of dialogue-summary pairs."""
        inputs = tokenizer(
            [config.prefix + d for d in batch[dlg_col]],
            max_length=config.max_src_length,
            truncation=True,
        )
        targets = tokenizer(
            text_target=batch["summary"],
            max_length=config.max_tgt_length,
            truncation=True,
        )
        # Replace pad token ids with -100 so they are ignored in the loss
        inputs["labels"] = [
            [(tok_id if tok_id != tokenizer.pad_token_id else -100) for tok_id in label_ids]
            for label_ids in targets["input_ids"]
        ]
        return inputs

    original_columns = ds["train"].column_names

    train_ds = (
        ds["train"]
        .select(range(min(config.n_train, len(ds["train"]))))
        .map(tokenize, batched=True, remove_columns=original_columns)
    )
    eval_ds = (
        ds["validation"]
        .select(range(min(config.n_eval, len(ds["validation"]))))
        .map(tokenize, batched=True, remove_columns=original_columns)
    )
    test_ds = ds["test"].select(range(min(config.n_test, len(ds["test"]))))

    logger.info(
        "Dataset ready — train=%d, eval=%d, test=%d",
        len(train_ds),
        len(eval_ds),
        len(test_ds),
    )

    return {
        "train": train_ds,
        "eval": eval_ds,
        "test": test_ds,
        "dialogue_column": dlg_col,
    }
