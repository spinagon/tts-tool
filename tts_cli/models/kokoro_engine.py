"""Kokoro-82M TTS Engine implementation."""

import os
import re
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple
import numpy as np

from ..audio_utils import save_audio
from ..downloader import ModelDownloader, get_model_cache_dir
from ..timestamps import (
    SentenceTimestamp,
    TimestampResult,
    WordTimestamp,
    align_phonemes_to_words,
    interpolate_word_timestamps,
)
from .base import BaseTTSEngine


VOICE_PRESETS = [
    # American English (Female)
    {"id": "af_heart", "name": "Heart (Default)", "lang": "en-us", "gender": "female"},
    {"id": "af_bella", "name": "Bella", "lang": "en-us", "gender": "female"},
    {"id": "af_nicole", "name": "Nicole", "lang": "en-us", "gender": "female"},
    {"id": "af_sarah", "name": "Sarah", "lang": "en-us", "gender": "female"},
    {"id": "af_sky", "name": "Sky", "lang": "en-us", "gender": "female"},
    {"id": "af_alloy", "name": "Alloy", "lang": "en-us", "gender": "female"},
    {"id": "af_aoede", "name": "Aoede", "lang": "en-us", "gender": "female"},
    {"id": "af_jessica", "name": "Jessica", "lang": "en-us", "gender": "female"},
    {"id": "af_kore", "name": "Kore", "lang": "en-us", "gender": "female"},
    # American English (Male)
    {"id": "am_adam", "name": "Adam", "lang": "en-us", "gender": "male"},
    {"id": "am_echo", "name": "Echo", "lang": "en-us", "gender": "male"},
    {"id": "am_eric", "name": "Eric", "lang": "en-us", "gender": "male"},
    {"id": "am_fenrir", "name": "Fenrir", "lang": "en-us", "gender": "male"},
    {"id": "am_liam", "name": "Liam", "lang": "en-us", "gender": "male"},
    {"id": "am_michael", "name": "Michael", "lang": "en-us", "gender": "male"},
    {"id": "am_onyx", "name": "Onyx", "lang": "en-us", "gender": "male"},
    {"id": "am_puck", "name": "Puck", "lang": "en-us", "gender": "male"},
    {"id": "am_santa", "name": "Santa", "lang": "en-us", "gender": "male"},
    # British English (Female)
    {"id": "bf_alice", "name": "Alice", "lang": "en-gb", "gender": "female"},
    {"id": "bf_emma", "name": "Emma", "lang": "en-gb", "gender": "female"},
    {"id": "bf_isabella", "name": "Isabella", "lang": "en-gb", "gender": "female"},
    {"id": "bf_lily", "name": "Lily", "lang": "en-gb", "gender": "female"},
    # British English (Male)
    {"id": "bm_daniel", "name": "Daniel", "lang": "en-gb", "gender": "male"},
    {"id": "bm_fable", "name": "Fable", "lang": "en-gb", "gender": "male"},
    {"id": "bm_george", "name": "George", "lang": "en-gb", "gender": "male"},
    {"id": "bm_lewis", "name": "Lewis", "lang": "en-gb", "gender": "male"},
    # Spanish
    {"id": "ef_dora", "name": "Dora", "lang": "es", "gender": "female"},
    {"id": "em_alex", "name": "Alex", "lang": "es", "gender": "male"},
    {"id": "em_santa", "name": "Santa", "lang": "es", "gender": "male"},
    # French
    {"id": "ff_siwis", "name": "Siwis", "lang": "fr-fr", "gender": "female"},
    # Hindi
    {"id": "hf_alpha", "name": "Alpha", "lang": "hi", "gender": "female"},
    {"id": "hf_beta", "name": "Beta", "lang": "hi", "gender": "female"},
    {"id": "hm_omega", "name": "Omega", "lang": "hi", "gender": "male"},
    {"id": "hm_psi", "name": "Psi", "lang": "hi", "gender": "male"},
    # Italian
    {"id": "if_sara", "name": "Sara", "lang": "it", "gender": "female"},
    {"id": "im_nicola", "name": "Nicola", "lang": "it", "gender": "male"},
    # Japanese
    {"id": "jf_alpha", "name": "Alpha", "lang": "ja", "gender": "female"},
    {"id": "jf_gongitsune", "name": "Gongitsune", "lang": "ja", "gender": "female"},
    {"id": "jf_nezumi", "name": "Nezumi", "lang": "ja", "gender": "female"},
    {"id": "jf_tebukuro", "name": "Tebukuro", "lang": "ja", "gender": "female"},
    {"id": "jm_kumo", "name": "Kumo", "lang": "ja", "gender": "male"},
    # Portuguese
    {"id": "pf_dora", "name": "Dora", "lang": "pt-br", "gender": "female"},
    {"id": "pm_alex", "name": "Alex", "lang": "pt-br", "gender": "male"},
    {"id": "pm_santa", "name": "Santa", "lang": "pt-br", "gender": "male"},
    # Mandarin
    {"id": "zf_xiaobei", "name": "Xiaobei", "lang": "cmn", "gender": "female"},
    {"id": "zf_xiaoni", "name": "Xiaoni", "lang": "cmn", "gender": "female"},
    {"id": "zf_xiaoxiao", "name": "Xiaoxiao", "lang": "cmn", "gender": "female"},
    {"id": "zf_xiaoyi", "name": "Xiaoyi", "lang": "cmn", "gender": "female"},
    {"id": "zm_yunjian", "name": "Yunjian", "lang": "cmn", "gender": "male"},
    {"id": "zm_yunxi", "name": "Yunxi", "lang": "cmn", "gender": "male"},
    {"id": "zm_yunxia", "name": "Yunxia", "lang": "cmn", "gender": "male"},
    {"id": "zm_yunyang", "name": "Yunyang", "lang": "cmn", "gender": "male"},
]


class KokoroEngine(BaseTTSEngine):
    """Kokoro-82M Text-to-Speech Engine."""

    name = "kokoro"
    description = "Kokoro-82M: Ultra-fast, lightweight high-fidelity TTS (CPU & GPU friendly)"

    def __init__(self):
        self._kokoro_instance = None
        self._kokoro_ts_instance = None
        self._pipeline_instance = None

    def is_available(self) -> bool:
        """Check if ONNX or PyTorch Kokoro runtime packages are installed."""
        try:
            import kokoro_onnx  # noqa: F401
            return True
        except ImportError:
            pass
        try:
            import kokoro  # noqa: F401
            return True
        except ImportError:
            pass
        return False

    def list_voices(self) -> List[Dict[str, str]]:
        return VOICE_PRESETS

    def preload(self, **kwargs) -> bool:
        """Download model weights and voice files into local cache."""
        try:
            ModelDownloader.preload_kokoro()
            return True
        except Exception as e:
            print(f"[Error preloading Kokoro]: {e}", file=sys.stderr)
            return False

    def get_supported_options(self) -> Dict[str, Dict[str, Any]]:
        return {
            "voice": {
                "type": "str",
                "default": "af_heart",
                "help": "Voice identifier (e.g. af_heart, af_bella, am_adam, bf_emma, bm_george)",
            },
            "speed": {
                "type": "float",
                "default": 1.0,
                "help": "Speech speed rate multiplier (0.5 to 2.0)",
            },
            "lang_code": {
                "type": "str",
                "default": "auto",
                "help": "Language code: 'en-us', 'en-gb', 'ja', 'cmn', 'es', 'fr-fr', 'hi', 'it', 'pt-br', or 'auto'",
            },
            "backend": {
                "type": "str",
                "default": "onnx",
                "choices": ["onnx", "pytorch"],
                "help": "Execution runtime backend ('onnx' or 'pytorch')",
            },
            "sentence_pause": {
                "type": "float",
                "default": 0.25,
                "help": "Pause duration in seconds between sentences (ONNX backend)",
            },
            "clause_pause": {
                "type": "float",
                "default": 0.1,
                "help": "Pause duration in seconds between clauses (ONNX backend)",
            },
            "continuous": {
                "type": "bool",
                "default": False,
                "help": "Synthesize with continuous overlapping windows for smoother prosody (ONNX backend)",
            },
            "trim": {
                "type": "bool",
                "default": True,
                "help": "Trim silence from start and end of generated audio",
            },
            "split_pattern": {
                "type": "str",
                "default": r"\n+",
                "help": "Regex pattern for text splitting (PyTorch backend)",
            },
        }

    def _infer_lang_code(self, voice: str, explicit_lang: Optional[str] = None, backend: str = "onnx") -> str:
        lang = explicit_lang.lower().strip() if explicit_lang and explicit_lang != "auto" else ""
        if not lang:
            prefix = voice[:1].lower()
            onnx_map = {
                "a": "en-us",
                "b": "en-gb",
                "j": "ja",
                "z": "cmn",
                "e": "es",
                "f": "fr-fr",
                "h": "hi",
                "i": "it",
                "p": "pt-br",
            }
            pt_map = {
                "a": "a",
                "b": "b",
                "j": "j",
                "z": "z",
                "e": "e",
                "f": "f",
                "h": "h",
                "i": "i",
                "p": "p",
            }
            return onnx_map.get(prefix, "en-us") if backend == "onnx" else pt_map.get(prefix, "a")

        if backend == "onnx":
            short_to_onnx = {
                "a": "en-us",
                "en": "en-us",
                "en-us": "en-us",
                "b": "en-gb",
                "en-gb": "en-gb",
                "j": "ja",
                "ja": "ja",
                "jp": "ja",
                "z": "cmn",
                "zh": "cmn",
                "cmn": "cmn",
                "e": "es",
                "es": "es",
                "f": "fr-fr",
                "fr": "fr-fr",
                "fr-fr": "fr-fr",
                "h": "hi",
                "hi": "hi",
                "i": "it",
                "it": "it",
                "p": "pt-br",
                "pt": "pt-br",
                "pt-br": "pt-br",
            }
            return short_to_onnx.get(lang, lang)
        else:
            onnx_to_pt = {
                "en-us": "a",
                "en-gb": "b",
                "ja": "j",
                "cmn": "z",
                "zh": "z",
                "es": "e",
                "fr-fr": "f",
                "fr": "f",
                "hi": "h",
                "it": "i",
                "pt-br": "p",
                "pt": "p",
            }
            return onnx_to_pt.get(lang, lang[:1])

    def _get_onnx_model(self, timestamped: bool = False):
        from kokoro_onnx import Kokoro

        kokoro_dir = get_model_cache_dir("kokoro")
        voices_path = kokoro_dir / "voices-v1.0.bin"
        if not voices_path.exists():
            ModelDownloader.preload_kokoro()

        if timestamped:
            ts_model = kokoro_dir / "kokoro-v1.0-timestamped.onnx"
            if not ts_model.exists():
                ModelDownloader.preload_kokoro_timestamped()
            if self._kokoro_ts_instance is None:
                self._kokoro_ts_instance = Kokoro(str(ts_model), str(voices_path))
                self._kokoro_ts_instance.has_timings = True
            return self._kokoro_ts_instance
        else:
            model_path = kokoro_dir / "kokoro-v1.0.onnx"
            if not model_path.exists():
                ModelDownloader.preload_kokoro()
            if self._kokoro_instance is None:
                self._kokoro_instance = Kokoro(str(model_path), str(voices_path))
            return self._kokoro_instance

    def _synthesize_with_timestamps(
        self,
        text: str,
        voice: str,
        speed: float,
        lang_code: str,
        sentence_pause: float = 0.25,
        clause_pause: float = 0.1,
        continuous: bool = False,
        trim: bool = True,
    ) -> Tuple[np.ndarray, TimestampResult]:
        """Synthesize and compute sentence & word timestamps using duration tensors."""
        model = self._get_onnx_model(timestamped=True)

        # Segment into sentences
        raw_sentences = [s.strip() for s in re.split(r"(?<=[.!?])\s+", text) if s.strip()]
        if not raw_sentences:
            raw_sentences = [text.strip()]

        audio_parts = []
        sentence_timings: List[SentenceTimestamp] = []
        all_word_timings: List[WordTimestamp] = []
        current_time = 0.0

        for idx, sentence in enumerate(raw_sentences):
            samples, sr, spoken = model.create_timed(
                sentence,
                voice=voice,
                speed=speed,
                lang=lang_code,
                sentence_pause=sentence_pause,
                clause_pause=clause_pause,
                continuous=continuous,
                trim=trim,
            )

            sent_duration = len(samples) / float(sr)
            sent_start = current_time
            sent_end = current_time + sent_duration

            # Extract words from phonemes or interpolate
            words = align_phonemes_to_words(sentence, spoken, sentence_offset=sent_start)
            if not words:
                words = interpolate_word_timestamps(sentence, sent_start, sent_end)

            sent_obj = SentenceTimestamp(
                sentence=sentence,
                start=sent_start,
                end=sent_end,
                words=words,
            )
            sentence_timings.append(sent_obj)
            all_word_timings.extend(words)

            audio_parts.append(samples)
            current_time = sent_end

            # Add inter-sentence pause if not last
            if idx < len(raw_sentences) - 1 and sentence_pause > 0:
                pause_samples = np.zeros(int(sentence_pause * sr), dtype=np.float32)
                audio_parts.append(pause_samples)
                current_time += sentence_pause

        combined_audio = np.concatenate(audio_parts) if audio_parts else np.zeros(2400, dtype=np.float32)
        result = TimestampResult(
            sentences=sentence_timings,
            words=all_word_timings,
            total_duration=current_time,
        )
        return combined_audio, result

    def synthesize(
        self,
        text: str,
        output_path: str,
        voice: str = "af_heart",
        speed: float = 1.0,
        lang_code: str = "auto",
        backend: str = "onnx",
        sentence_pause: float = 0.25,
        clause_pause: float = 0.1,
        continuous: bool = False,
        trim: bool = True,
        split_pattern: str = r"\n+",
        timestamps: bool = False,
        **kwargs: Any,
    ) -> Tuple[str, Optional[TimestampResult]]:
        """Convert text to speech using Kokoro."""
        lang = self._infer_lang_code(voice, lang_code, backend=backend)

        if timestamps:
            audio_data, ts_result = self._synthesize_with_timestamps(
                text=text,
                voice=voice,
                speed=speed,
                lang_code=lang,
                sentence_pause=sentence_pause,
                clause_pause=clause_pause,
                continuous=continuous,
                trim=trim,
            )
            saved_file = save_audio(audio_data, output_path, sample_rate=24000)
            return saved_file, ts_result

        # Standard fast path (no timestamps)
        model = self._get_onnx_model(timestamped=False)
        audio_data, _ = model.create(
            text,
            voice=voice,
            speed=speed,
            lang=lang,
            sentence_pause=sentence_pause,
            clause_pause=clause_pause,
            continuous=continuous,
            trim=trim,
        )
        saved_file = save_audio(audio_data, output_path, sample_rate=24000)
        return saved_file, None
