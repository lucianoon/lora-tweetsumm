"""
Configuration management for the LoRA fine-tuning pipeline.

Loads hyperparameters from a YAML file and exposes them as a typed dataclass,
with automatic device detection (MPS → CUDA → CPU).
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from pathlib import Path

import torch
import yaml

logger = logging.getLogger(__name__)

_PROJECT_ROOT = Path(__file__).resolve().parent.parent
_DEFAULT_CONFIG = _PROJECT_ROOT / "configs" / "default.yaml"


@dataclass
class LoraParams:
    """LoRA adapter hyperparameters."""

    r: int = 8
    alpha: int = 16
    dropout: float = 0.05
    target_modules: list[str] = field(default_factory=lambda: ["q", "v"])
    use_rslora: bool = True


@dataclass
class TrainingParams:
    """Seq2Seq training hyperparameters."""

    epochs: int = 3
    batch_size: int = 4
    learning_rate: float = 1e-3
    logging_steps: int = 20
    eval_strategy: str = "epoch"
    bf16_on_cuda: bool = True
    output_dir: str = "checkpoints/t5-lora-tweetsumm"


@dataclass
class InferenceParams:
    """Generation parameters."""

    max_new_tokens: int = 64
    num_beams: int = 4


@dataclass
class Config:
    """Top-level configuration for the entire pipeline.

    Attributes:
        model_id: HuggingFace model identifier for the base T5 model.
        dataset_name: HuggingFace dataset identifier.
        n_train: Number of training samples to use (for fast iteration).
        n_eval: Number of evaluation samples.
        n_test: Number of test samples for final evaluation.
        max_src_length: Maximum source sequence length (tokens).
        max_tgt_length: Maximum target sequence length (tokens).
        prefix: Task prefix prepended to every input (T5-style).
        device: Compute device, auto-detected if not specified.
        lora: LoRA adapter configuration.
        training: Training loop configuration.
        inference: Generation configuration.
        results_dir: Directory to save evaluation results.
    """

    model_id: str = "google-t5/t5-small"
    dataset_name: str = "Andyrasika/TweetSumm-tuned"
    n_train: int = 300
    n_eval: int = 50
    n_test: int = 50
    max_src_length: int = 512
    max_tgt_length: int = 64
    prefix: str = "summarize: "
    device: str = ""
    lora: LoraParams = field(default_factory=LoraParams)
    training: TrainingParams = field(default_factory=TrainingParams)
    inference: InferenceParams = field(default_factory=InferenceParams)
    results_dir: str = "results"

    def __post_init__(self) -> None:
        if not self.device:
            self.device = _detect_device()
        logger.info("Config loaded — model=%s, device=%s", self.model_id, self.device)


def _detect_device() -> str:
    """Select the best available compute device."""
    if torch.backends.mps.is_available():
        return "mps"
    if torch.cuda.is_available():
        return "cuda"
    return "cpu"


def load_config(path: str | Path | None = None) -> Config:
    """Load configuration from a YAML file.

    Args:
        path: Path to the YAML config file. Defaults to ``configs/default.yaml``.

    Returns:
        A fully-initialized :class:`Config` instance.
    """
    config_path = Path(path) if path else _DEFAULT_CONFIG

    if not config_path.exists():
        logger.warning("Config file not found at %s — using defaults.", config_path)
        return Config()

    with open(config_path) as f:
        raw = yaml.safe_load(f) or {}

    lora_raw = raw.pop("lora", {})
    training_raw = raw.pop("training", {})
    inference_raw = raw.pop("inference", {})

    return Config(
        lora=LoraParams(**lora_raw),
        training=TrainingParams(**training_raw),
        inference=InferenceParams(**inference_raw),
        **raw,
    )
