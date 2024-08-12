"""
Inference utilities for generating summaries from dialogue input.

Provides functions for single-sample generation and before/after
comparison to demonstrate the effect of fine-tuning.
"""

from __future__ import annotations

import logging
from typing import Any, Dict

import torch
from transformers import PreTrainedModel, PreTrainedTokenizerBase

from src.config import Config

logger = logging.getLogger(__name__)


def summarize(
    model: PreTrainedModel,
    tokenizer: PreTrainedTokenizerBase,
    dialogue: str,
    config: Config,
) -> str:
    """Generate a summary for a single dialogue.

    Args:
        model: The (possibly fine-tuned) model.
        tokenizer: Corresponding tokenizer.
        dialogue: Raw dialogue text to summarize.
        config: Pipeline configuration (for prefix, max lengths, beam params).

    Returns:
        The generated summary as a string.
    """
    input_text = config.prefix + dialogue
    inputs = tokenizer(
        input_text,
        return_tensors="pt",
        truncation=True,
        max_length=config.max_src_length,
    ).to(model.device)

    with torch.inference_mode():
        outputs = model.generate(
            **inputs,
            max_new_tokens=config.inference.max_new_tokens,
            num_beams=config.inference.num_beams,
        )

    summary = tokenizer.decode(outputs[0], skip_special_tokens=True)
    return summary


def compare_before_after(
    model: PreTrainedModel,
    tokenizer: PreTrainedTokenizerBase,
    sample: Dict[str, Any],
    dialogue_column: str,
    config: Config,
) -> Dict[str, str]:
    """Generate and display a summary for a test sample.

    This is typically called twice — once before training and once after —
    to illustrate the improvement from fine-tuning.

    Args:
        model: The model (before or after training).
        tokenizer: Corresponding tokenizer.
        sample: A single test sample with dialogue and reference summary.
        dialogue_column: Name of the dialogue column in the dataset.
        config: Pipeline configuration.

    Returns:
        A dict with ``"generated"`` and ``"reference"`` summaries.
    """
    generated = summarize(model, tokenizer, sample[dialogue_column], config)
    reference = sample["summary"]

    return {
        "generated": generated,
        "reference": reference,
    }


def print_comparison(label: str, result: Dict[str, str]) -> None:
    """Pretty-print a before/after comparison result.

    Args:
        label: Label for the comparison (e.g., "BEFORE training").
        result: Output from :func:`compare_before_after`.
    """
    separator = "─" * 60
    print(f"\n{separator}")
    print(f"  {label}")
    print(separator)
    print(f"  Generated:  {result['generated']}")
    print(f"  Reference:  {result['reference']}")
    print(separator)
