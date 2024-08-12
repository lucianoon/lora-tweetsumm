"""Shared fixtures for the test suite."""

from __future__ import annotations

import pytest

from src.config import Config, InferenceParams, LoraParams, TrainingParams


@pytest.fixture
def sample_config() -> Config:
    """Create a minimal Config for testing (no file I/O, forced CPU)."""
    return Config(
        model_id="google-t5/t5-small",
        dataset_name="Andyrasika/TweetSumm-tuned",
        n_train=10,
        n_eval=5,
        n_test=5,
        max_src_length=128,
        max_tgt_length=32,
        prefix="summarize: ",
        device="cpu",
        lora=LoraParams(r=4, alpha=16, dropout=0.05),
        training=TrainingParams(epochs=1, batch_size=2),
        inference=InferenceParams(max_new_tokens=32, num_beams=1),
        results_dir="results",
    )


@pytest.fixture
def sample_dialogue() -> str:
    """Return a short dialogue for testing inference."""
    return (
        "Customer: Hi, I need to cancel my order #12345.\n"
        "Agent: Sure, I can help with that. Let me look it up.\n"
        "Agent: Your order has been cancelled. Refund in 3-5 business days.\n"
        "Customer: Thank you."
    )
