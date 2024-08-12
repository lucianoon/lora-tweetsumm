"""Tests for src.data — dataset loading and preprocessing."""

from __future__ import annotations

import pytest
from datasets import Dataset, DatasetDict

from src.data import _detect_dialogue_column


class TestDetectDialogueColumn:
    """Verify column name detection logic."""

    def test_detects_dialogue_column(self):
        mock_ds = DatasetDict(
            {
                "train": Dataset.from_dict(
                    {
                        "dialogue": ["Hello", "World"],
                        "summary": ["Hi", "Earth"],
                    }
                )
            }
        )
        assert _detect_dialogue_column(mock_ds) == "dialogue"

    def test_detects_conversation_column(self):
        mock_ds = DatasetDict(
            {
                "train": Dataset.from_dict(
                    {
                        "conversation": ["Hello", "World"],
                        "summary": ["Hi", "Earth"],
                    }
                )
            }
        )
        assert _detect_dialogue_column(mock_ds) == "conversation"

    def test_detects_dialog_column(self):
        mock_ds = DatasetDict(
            {
                "train": Dataset.from_dict(
                    {
                        "dialog": ["Hello", "World"],
                        "summary": ["Hi", "Earth"],
                    }
                )
            }
        )
        assert _detect_dialogue_column(mock_ds) == "dialog"

    def test_raises_on_missing_column(self):
        mock_ds = DatasetDict(
            {
                "train": Dataset.from_dict(
                    {
                        "text": ["Hello"],
                        "summary": ["Hi"],
                    }
                )
            }
        )
        with pytest.raises(ValueError, match="Could not find a dialogue column"):
            _detect_dialogue_column(mock_ds)

    def test_priority_order(self):
        """'dialogue' should be preferred over 'conversation' and 'dialog'."""
        mock_ds = DatasetDict(
            {
                "train": Dataset.from_dict(
                    {
                        "dialogue": ["A"],
                        "conversation": ["B"],
                        "dialog": ["C"],
                        "summary": ["S"],
                    }
                )
            }
        )
        assert _detect_dialogue_column(mock_ds) == "dialogue"
