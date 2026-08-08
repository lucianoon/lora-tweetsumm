"""Tests for src.train — training-argument construction.

These exercise the pure seam ``build_training_args(config)`` so the mapping
from config to :class:`Seq2SeqTrainingArguments` is verified without building a
real model (no download, runs in the fast CI suite).
"""

from __future__ import annotations

from src.config import Config, TrainingParams
from src.train import build_training_args


class TestBestCheckpointSelection:
    """The trainer should keep the best checkpoint, not the last epoch."""

    def test_restores_best_model_by_eval_loss(self, sample_config):
        args = build_training_args(sample_config)
        assert args.load_best_model_at_end is True
        assert args.metric_for_best_model == "eval_loss"


class TestWarmup:
    """The learning-rate warmup should be wired to the config, not hardcoded."""

    def test_warmup_steps_comes_from_config(self):
        config = Config(device="cpu", training=TrainingParams(warmup_steps=0.1))
        args = build_training_args(config)
        assert args.warmup_steps == 0.1


class TestBf16Guard:
    """bf16 must stay off outside CUDA, even when opted in."""

    def test_bf16_disabled_on_non_cuda(self):
        config = Config(device="mps", training=TrainingParams(bf16_on_cuda=True))
        args = build_training_args(config)
        assert args.bf16 is False
