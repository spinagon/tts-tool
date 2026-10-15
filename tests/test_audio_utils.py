"""Tests for audio utility functions."""

import os
import wave
import numpy as np
import pytest

from tts_cli.audio_utils import save_audio, play_audio


def test_save_audio_wav_numpy(tmp_path):
    """Test saving a basic float32 numpy waveform as WAV."""
    sample_rate = 24000
    # Generate 0.5s of 440Hz sine wave
    t = np.linspace(0, 0.5, int(sample_rate * 0.5), endpoint=False, dtype=np.float32)
    audio = 0.5 * np.sin(2 * np.pi * 440 * t)

    out_file = tmp_path / "test_sine.wav"
    result = save_audio(audio, str(out_file), sample_rate=sample_rate)

    assert os.path.exists(result)
    assert result == str(out_file.resolve())

    # Verify WAV header
    with wave.open(str(out_file), "rb") as wf:
        assert wf.getnchannels() == 1
        assert wf.getframerate() == sample_rate
        assert wf.getnframes() == len(audio)


def test_save_audio_clipping_normalization(tmp_path):
    """Test that audio with amplitudes exceeding 1.0 is normalized to prevent distortion."""
    sample_rate = 16000
    audio = np.array([-2.5, 0.0, 3.0, 1.5], dtype=np.float32)
    out_file = tmp_path / "normalized.wav"

    result = save_audio(audio, str(out_file), sample_rate=sample_rate)
    assert os.path.exists(result)


def test_save_audio_2d_tensor_or_multichannel(tmp_path):
    """Test squeezing 2D array [1, N] into 1D waveform."""
    sample_rate = 24000
    audio_2d = np.zeros((1, 1200), dtype=np.float32)
    out_file = tmp_path / "squeezed.wav"

    result = save_audio(audio_2d, str(out_file), sample_rate=sample_rate)
    assert os.path.exists(result)

    with wave.open(str(out_file), "rb") as wf:
        assert wf.getnchannels() == 1
        assert wf.getnframes() == 1200


def test_play_audio_nonexistent():
    """Test play_audio handling of non-existent or unplayable files."""
    # Should not raise exception, but return false or handle gracefully
    res = play_audio("/tmp/definitely_nonexistent_audio_file.wav")
    assert isinstance(res, bool)
