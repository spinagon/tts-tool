"""Tests for Command Line Interface (CLI)."""

import os
import sys
from pathlib import Path
import pytest

from tts_cli.cli import build_parser, read_input_text, run_synthesis


def test_build_parser_subcommands():
    """Test CLI parser structures and subcommands."""
    parser = build_parser()

    # 1. Kokoro
    args = parser.parse_args(["kokoro", "Hello world", "--voice", "bm_george", "--speed", "1.2"])
    assert args.subcommand == "kokoro"
    assert args.text == "Hello world"
    assert args.voice == "bm_george"
    assert args.speed == 1.2
    assert args.output == "output.wav"

    # 2. Kokoro with timestamps
    args_ts = parser.parse_args(["kokoro", "Sample text", "--timestamps", "srt", "--timestamps-level", "word"])
    assert args_ts.timestamps == "srt"
    assert args_ts.timestamps_level == "word"

    # 3. Chatterbox
    args_cb = parser.parse_args([
        "chatterbox", "Test text",
        "--variant", "turbo",
        "--reference-audio", "sample.wav",
        "--exaggeration", "0.8",
        "--cfg-weight", "0.6",
    ])
    assert args_cb.subcommand == "chatterbox"
    assert args_cb.variant == "turbo"
    assert args_cb.reference_audio == "sample.wav"
    assert args_cb.exaggeration == 0.8
    assert args_cb.cfg_weight == 0.6

    # 4. Fish Speech
    args_fish = parser.parse_args([
        "fish-speech", "Fish test",
        "--reference-audio", "ref.wav",
        "--prompt-text", "Transcript of ref",
        "--temperature", "0.65",
    ])
    assert args_fish.subcommand == "fish-speech"
    assert args_fish.reference_audio == "ref.wav"
    assert args_fish.prompt_text == "Transcript of ref"
    assert args_fish.temperature == 0.65

    # 5. Preload
    args_pre = parser.parse_args(["preload", "--engine", "kokoro"])
    assert args_pre.subcommand == "preload"
    assert args_pre.engine == "kokoro"

    # 6. Status & Voices
    assert parser.parse_args(["status"]).subcommand == "status"
    assert parser.parse_args(["list-voices", "--engine", "kokoro"]).subcommand == "list-voices"


def test_read_input_text(tmp_path):
    """Test reading input from args, flags, and files."""
    # Direct argument
    assert read_input_text("Hello arg", None) == "Hello arg"

    # From file
    sample_file = tmp_path / "input.txt"
    sample_file.write_text("Hello from file", encoding="utf-8")
    assert read_input_text(None, str(sample_file)) == "Hello from file"


def test_run_synthesis_end_to_end(tmp_path):
    """Test full run_synthesis execution with subtitle export."""
    out_wav = tmp_path / "cli_test.wav"
    out_result = run_synthesis(
        engine_name="kokoro",
        text="Testing automated regression test suite.",
        output_path=str(out_wav),
        play=False,
        timestamps="srt",
        timestamps_level="both",
    )

    assert os.path.exists(out_result)
    assert os.path.getsize(out_result) > 1000

    # Verify generated subtitle files
    srt_file = tmp_path / "cli_test.srt"
    words_srt_file = tmp_path / "cli_test.words.srt"
    assert srt_file.exists()
    assert words_srt_file.exists()
    assert "Testing automated regression test suite." in srt_file.read_text(encoding="utf-8")
