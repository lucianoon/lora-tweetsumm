"""Tests for src.inference — summary generation utilities.

These tests load the actual T5-small model and are marked as slow.
"""

from __future__ import annotations

import pytest

from src.inference import compare_before_after, summarize
from src.model import build_model


@pytest.mark.slow
class TestSummarize:
    """Integration tests for the summarize function."""

    @pytest.fixture(autouse=True)
    def _setup(self, sample_config, sample_dialogue):
        from src.data import load_tokenizer

        self.config = sample_config
        self.tokenizer = load_tokenizer(self.config)
        self.model = build_model(self.config)
        self.dialogue = sample_dialogue

    def test_returns_string(self):
        result = summarize(self.model, self.tokenizer, self.dialogue, self.config)
        assert isinstance(result, str)

    def test_returns_non_empty(self):
        result = summarize(self.model, self.tokenizer, self.dialogue, self.config)
        assert len(result.strip()) > 0

    def test_output_differs_from_input(self):
        """The summary should not be a verbatim copy of the input."""
        result = summarize(self.model, self.tokenizer, self.dialogue, self.config)
        assert result != self.dialogue


@pytest.mark.slow
class TestCompareBeforeAfter:
    """Tests for the compare_before_after utility."""

    @pytest.fixture(autouse=True)
    def _setup(self, sample_config, sample_dialogue):
        from src.data import load_tokenizer

        self.config = sample_config
        self.tokenizer = load_tokenizer(self.config)
        self.model = build_model(self.config)
        self.sample = {"dialogue": sample_dialogue, "summary": "Order cancelled with refund."}

    def test_returns_expected_keys(self):
        result = compare_before_after(
            self.model, self.tokenizer, self.sample, "dialogue", self.config
        )
        assert "generated" in result
        assert "reference" in result

    def test_reference_matches_input(self):
        result = compare_before_after(
            self.model, self.tokenizer, self.sample, "dialogue", self.config
        )
        assert result["reference"] == "Order cancelled with refund."

    def test_generated_is_string(self):
        result = compare_before_after(
            self.model, self.tokenizer, self.sample, "dialogue", self.config
        )
        assert isinstance(result["generated"], str)
        assert len(result["generated"]) > 0
