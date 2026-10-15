"""Tests for model downloader and cache manager."""

import os
from pathlib import Path
import pytest

from tts_cli.downloader import (
    ModelDownloader,
    format_size,
    get_cache_root,
    get_model_cache_dir,
)


def test_format_size():
    """Test human-readable byte sizing formatter."""
    assert "B" in format_size(500)
    assert "KB" in format_size(2048)
    assert "MB" in format_size(1024 * 1024 * 5)
    assert "GB" in format_size(1024 * 1024 * 1024 * 2)


def test_cache_directories(tmp_path, monkeypatch):
    """Test cache directory resolution and custom env var overrides."""
    # Custom env var override
    custom_dir = tmp_path / "custom_tts_cache"
    monkeypatch.setenv("TTS_CACHE_DIR", str(custom_dir))

    root = get_cache_root()
    assert root == custom_dir
    assert root.exists()

    kokoro_dir = get_model_cache_dir("kokoro")
    assert kokoro_dir == custom_dir / "models" / "kokoro"
    assert kokoro_dir.exists()

    chatterbox_dir = get_model_cache_dir("chatterbox")
    assert chatterbox_dir.exists()

    fish_dir = get_model_cache_dir("fish_speech")
    assert fish_dir.exists()


def test_downloader_get_status():
    """Test status query structure."""
    status = ModelDownloader.get_status()
    assert "kokoro" in status
    assert "chatterbox" in status
    assert "fish-speech" in status

    for eng, info in status.items():
        assert "status" in info
        assert "path" in info
        assert "size" in info
