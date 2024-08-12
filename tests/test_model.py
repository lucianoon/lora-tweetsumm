"""Tests for src.model — model construction and statistics.

These tests load the actual T5-small model (~240MB) and are marked as slow.
Run with: pytest -m slow
"""

from __future__ import annotations

import pytest
import torch

from src.model import build_model, find_latest_checkpoint, get_model_stats, read_adapter_rank


class TestFindLatestCheckpoint:
    """Tests for checkpoint discovery without loading model weights."""

    def test_returns_highest_numeric_checkpoint_with_adapter_config(self, tmp_path):
        output_dir = tmp_path / "checkpoints"
        (output_dir / "checkpoint-10").mkdir(parents=True)
        (output_dir / "checkpoint-10" / "adapter_config.json").write_text("{}")
        (output_dir / "checkpoint-200").mkdir()
        (output_dir / "checkpoint-200" / "adapter_config.json").write_text("{}")
        (output_dir / "checkpoint-50").mkdir()
        (output_dir / "checkpoint-50" / "adapter_config.json").write_text("{}")

        assert find_latest_checkpoint(output_dir) == output_dir / "checkpoint-200"

    def test_ignores_invalid_checkpoint_directories(self, tmp_path):
        output_dir = tmp_path / "checkpoints"
        (output_dir / "checkpoint-final").mkdir(parents=True)
        (output_dir / "checkpoint-final" / "adapter_config.json").write_text("{}")
        (output_dir / "checkpoint-20").mkdir()
        (output_dir / "checkpoint-20" / "adapter_config.json").write_text("{}")
        (output_dir / "checkpoint-30").mkdir()

        assert find_latest_checkpoint(output_dir) == output_dir / "checkpoint-20"

    def test_raises_when_no_adapter_checkpoint_exists(self, tmp_path):
        with pytest.raises(FileNotFoundError, match="No LoRA adapter checkpoints"):
            find_latest_checkpoint(tmp_path / "missing")


class TestReadAdapterRank:
    """Tests for PEFT adapter metadata parsing."""

    def test_reads_rank_from_adapter_config(self, tmp_path):
        checkpoint = tmp_path / "checkpoint-1"
        checkpoint.mkdir()
        (checkpoint / "adapter_config.json").write_text('{"r": 4}')

        assert read_adapter_rank(checkpoint) == 4

    def test_returns_none_for_missing_or_invalid_rank(self, tmp_path):
        checkpoint = tmp_path / "checkpoint-1"
        checkpoint.mkdir()
        (checkpoint / "adapter_config.json").write_text('{"r": "4"}')

        assert read_adapter_rank(checkpoint) is None
        assert read_adapter_rank(tmp_path / "missing") is None


class TestGetModelStatsNoDownload:
    """Unit tests for model stats that avoid loading real model weights."""

    def test_counts_frozen_lora_params_for_loaded_adapter(self):
        class DummyPeftModel(torch.nn.Module):
            peft_config = {"default": object()}

            def __init__(self):
                super().__init__()
                self.base_weight = torch.nn.Parameter(torch.zeros(5), requires_grad=False)
                self.lora_A = torch.nn.Parameter(torch.zeros(2, 3), requires_grad=False)
                self.lora_B = torch.nn.Parameter(torch.zeros(3, 2), requires_grad=False)

            def get_nb_trainable_parameters(self):
                return 0, 17

        stats = get_model_stats(DummyPeftModel())

        assert stats["trainable_params"] == 12
        assert stats["total_params"] == 17
        assert stats["trainable_pct"] == round(100 * 12 / 17, 4)


@pytest.mark.slow
class TestBuildModel:
    """Integration tests that load the real T5-small model."""

    @pytest.fixture(autouse=True)
    def _build(self, sample_config):
        """Build model once per test class."""
        self.config = sample_config
        self.model = build_model(self.config)

    def test_model_has_lora_adapters(self):
        """Model should be wrapped with PEFT."""
        assert hasattr(self.model, "peft_config")
        assert hasattr(self.model, "get_nb_trainable_parameters")

    def test_trainable_params_less_than_total(self):
        trainable, total = self.model.get_nb_trainable_parameters()
        assert trainable < total
        assert trainable > 0

    def test_trainable_percentage_under_five(self):
        """LoRA should train < 5% of parameters."""
        trainable, total = self.model.get_nb_trainable_parameters()
        pct = 100 * trainable / total
        assert pct < 5.0

    def test_model_on_correct_device(self):
        device = str(next(self.model.parameters()).device)
        assert device == self.config.device


@pytest.mark.slow
class TestGetModelStats:
    """Tests for the get_model_stats utility."""

    @pytest.fixture(autouse=True)
    def _build(self, sample_config):
        self.model = build_model(sample_config)
        self.stats = get_model_stats(self.model)

    def test_returns_expected_keys(self):
        expected = {"total_params", "trainable_params", "trainable_pct", "model_size_mb"}
        assert set(self.stats.keys()) == expected

    def test_values_are_positive(self):
        assert self.stats["total_params"] > 0
        assert self.stats["trainable_params"] > 0
        assert self.stats["trainable_pct"] > 0
        assert self.stats["model_size_mb"] > 0

    def test_trainable_pct_is_reasonable(self):
        assert 0 < self.stats["trainable_pct"] < 5.0

    def test_model_size_is_reasonable(self):
        # T5-small is ~240MB in float32
        assert 50 < self.stats["model_size_mb"] < 500
