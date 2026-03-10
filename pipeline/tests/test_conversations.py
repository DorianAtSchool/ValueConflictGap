"""Tests for conversations.py — conversation structure and saving."""

import sys, os, json, tempfile
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from unittest.mock import MagicMock, patch
from conversations import save_conversation


def test_save_conversation(tmp_path):
    conv = [
        {"role": "user", "content": "Hello"},
        {"role": "assistant", "content": "Hi there!"},
        {"role": "user", "content": "How are you?"},
        {"role": "assistant", "content": "I'm well."},
    ]

    with patch("conversations.RESULTS_DIR", tmp_path):
        path = save_conversation(conv, "sarcasm", "HHH", "politics", 5)

    assert path.exists()
    with open(path) as f:
        loaded = json.load(f)
    assert len(loaded) == 4
    assert loaded[0]["role"] == "user"
    assert loaded[1]["role"] == "assistant"


def test_save_conversation_creates_dirs(tmp_path):
    conv = [{"role": "user", "content": "test"}]
    with patch("conversations.RESULTS_DIR", tmp_path):
        path = save_conversation(conv, "humor", "modelspec", "therapy", 10)
    assert "humor" in str(path)
    assert "modelspec" in str(path)
    assert path.exists()
