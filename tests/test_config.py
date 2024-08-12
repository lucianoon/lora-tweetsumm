"""Tests for src.config — configuration loading and defaults."""

from __future__ import annotations

from src.config import (
    Config,
    InferenceParams,
    LoraParams,
    TrainingParams,
    _detect_device,
    load_config,
)


class TestDataclassDefaults:
    """Verify that dataclass defaults are sensible."""

    def test_lora_params_defaults(self):
        lora = LoraParams()
        assert lora.r == 8
        assert lora.alpha == 16
        assert lora.dropout == 0.05
        assert lora.target_modules == ["q", "v"]
        assert lora.use_rslora is True

    def test_training_params_defaults(self):
        tp = TrainingParams()
        assert tp.epochs == 3
        assert tp.batch_size == 4
        assert tp.learning_rate == 1e-3
        assert tp.eval_strategy == "epoch"

    def test_inference_params_defaults(self):
        ip = InferenceParams()
        assert ip.max_new_tokens == 64
        assert ip.num_beams == 4

    def test_config_defaults(self):
        cfg = Config(device="cpu")
        assert cfg.model_id == "google-t5/t5-small"
        assert cfg.n_train == 300
        assert cfg.prefix == "summarize: "
        assert isinstance(cfg.lora, LoraParams)
        assert isinstance(cfg.training, TrainingParams)
        assert isinstance(cfg.inference, InferenceParams)


class TestDeviceDetection:
    """Verify device detection returns a valid string."""

    def test_returns_valid_device(self):
        device = _detect_device()
        assert device in ("mps", "cuda", "cpu")


class TestLoadConfig:
    """Verify YAML config loading."""

    def test_load_default_config(self):
        cfg = load_config()
        assert cfg.model_id == "google-t5/t5-small"
        assert isinstance(cfg.lora.r, int)
        assert cfg.lora.r > 0

    def test_load_fast_config(self):
        cfg = load_config("configs/fast.yaml")
        assert cfg.n_train == 100
        assert cfg.training.epochs == 1

    def test_load_missing_file_uses_defaults(self, tmp_path):
        cfg = load_config(tmp_path / "nonexistent.yaml")
        assert cfg.model_id == "google-t5/t5-small"

    def test_config_overrides_from_yaml(self, tmp_path):
        custom_yaml = tmp_path / "custom.yaml"
        custom_yaml.write_text("model_id: google-t5/t5-base\nn_train: 50\nlora:\n  r: 16\n")
        cfg = load_config(custom_yaml)
        assert cfg.model_id == "google-t5/t5-base"
        assert cfg.n_train == 50
        assert cfg.lora.r == 16
        # Non-overridden values keep defaults
        assert cfg.lora.alpha == 16
        assert cfg.training.epochs == 3
