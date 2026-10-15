"""Tests for TTS model engines."""

import os
from pathlib import Path
import pytest

from tts_cli.models import (
    ChatterboxEngine,
    FishSpeechEngine,
    KokoroEngine,
    get_engine,
    list_supported_engines,
)


def test_engine_registry():
    """Test engine lookup and aliases."""
    engines = list_supported_engines()
    assert "kokoro" in engines
    assert "chatterbox" in engines
    assert "fish-speech" in engines

    # Lookup canonical names
    assert isinstance(get_engine("kokoro"), KokoroEngine)
    assert isinstance(get_engine("chatterbox"), ChatterboxEngine)
    assert isinstance(get_engine("fish-speech"), FishSpeechEngine)

    # Aliases
    assert isinstance(get_engine("fish"), FishSpeechEngine)
    assert isinstance(get_engine("fish_speech"), FishSpeechEngine)

    # Invalid engine
    with pytest.raises(ValueError):
        get_engine("non_existent_engine")


def test_kokoro_options_and_voices():
    """Test Kokoro voice listing and language inference."""
    eng = KokoroEngine()
    voices = eng.list_voices()
    assert len(voices) > 20

    voice_ids = {v["id"] for v in voices}
    assert "af_heart" in voice_ids
    assert "bm_george" in voice_ids
    assert "zf_xiaobei" in voice_ids

    # Test language inference
    assert eng._infer_lang_code("af_heart") == "en-us"
    assert eng._infer_lang_code("bf_emma") == "en-gb"
    assert eng._infer_lang_code("jf_alpha") == "ja"
    assert eng._infer_lang_code("zf_xiaobei") == "cmn"
    assert eng._infer_lang_code("ef_dora") == "es"
    assert eng._infer_lang_code("ff_siwis") == "fr-fr"

    # Explicit override
    assert eng._infer_lang_code("af_heart", explicit_lang="en-gb") == "en-gb"


def test_chatterbox_options_and_device():
    """Test Chatterbox options and device resolver."""
    eng = ChatterboxEngine()
    opts = eng.get_supported_options()
    assert "variant" in opts
    assert "exaggeration" in opts
    assert "cfg_weight" in opts

    dev = eng._resolve_device("cpu")
    assert dev == "cpu"


def test_fish_speech_options():
    """Test Fish Speech options."""
    eng = FishSpeechEngine()
    opts = eng.get_supported_options()
    assert "prompt_text" in opts
    assert "temperature" in opts
    assert "repetition_penalty" in opts


def test_kokoro_synthesis_and_timestamps(tmp_path):
    """End-to-end test of Kokoro synthesis with timestamps."""
    eng = KokoroEngine()
    if not eng.is_available():
        pytest.skip("Kokoro runtime dependencies not installed.")

    # 1. Basic synthesis
    out_file = tmp_path / "test_kokoro.wav"
    result_path, ts = eng.synthesize(
        text="Fast test.",
        output_path=str(out_file),
        voice="af_heart",
        timestamps=False,
    )
    assert os.path.exists(result_path)
    assert ts is None
    assert os.path.getsize(result_path) > 1000

    # 2. Synthesis with timestamps
    out_ts_file = tmp_path / "test_kokoro_ts.wav"
    result_path_ts, ts_result = eng.synthesize(
        text="First sentence. Second sentence.",
        output_path=str(out_ts_file),
        voice="af_heart",
        timestamps=True,
    )
    assert os.path.exists(result_path_ts)
    assert ts_result is not None
    assert len(ts_result.sentences) == 2
    assert len(ts_result.words) >= 4
    assert ts_result.total_duration > 0.5
