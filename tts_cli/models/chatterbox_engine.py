"""Chatterbox TTS Engine implementation (Resemble AI)."""

import os
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple
import numpy as np

from ..audio_utils import save_audio
from ..downloader import ModelDownloader
from .base import BaseTTSEngine


def _ensure_chatterbox_imported() -> bool:
    """Attempt to import chatterbox, checking local and project virtualenvs if needed."""
    try:
        import chatterbox  # noqa: F401
        return True
    except ImportError:
        pass

    candidates = [
        Path.home() / "prog" / "tts-chatterbox" / ".venv" / "lib",
        Path.home() / "temp" / "scratch" / ".venv" / "lib",
    ]
    for c_dir in candidates:
        if c_dir.exists():
            for sp in c_dir.glob("python*/site-packages"):
                if str(sp) not in sys.path:
                    sys.path.insert(0, str(sp))
                    try:
                        import chatterbox  # noqa: F401
                        return True
                    except ImportError:
                        pass
    return False


class ChatterboxEngine(BaseTTSEngine):
    """Chatterbox TTS Engine supporting Standard, Turbo, and Multilingual variants with Zero-Shot Cloning."""

    name = "chatterbox"
    description = "Chatterbox (Resemble AI): Expressive voice cloning, emotional exaggeration, and paralinguistic tags"

    def __init__(self):
        self._cached_models: Dict[str, Any] = {}

    def is_available(self) -> bool:
        """Check if chatterbox-tts package is installed."""
        return _ensure_chatterbox_imported()

    def list_voices(self) -> List[Dict[str, str]]:
        return [
            {
                "id": "default",
                "name": "Default Resemble Speaker",
                "lang": "en",
                "gender": "neutral",
                "description": "Standard built-in reference voice",
            },
            {
                "id": "custom",
                "name": "Zero-Shot Reference Voice",
                "lang": "any",
                "gender": "variable",
                "description": "Use --reference-audio <file.wav> to clone any voice",
            },
        ]

    def preload(self, variant: str = "standard", **kwargs) -> bool:
        """Preload model weights for specified Chatterbox variant."""
        try:
            ModelDownloader.preload_chatterbox(variant=variant)
            return True
        except Exception as e:
            print(f"[Error preloading Chatterbox ({variant})]: {e}", file=sys.stderr)
            return False

    def get_supported_options(self) -> Dict[str, Dict[str, Any]]:
        return {
            "variant": {
                "type": "str",
                "default": "standard",
                "choices": ["standard", "turbo", "multilingual"],
                "help": "Chatterbox variant: 'standard' (0.5B expressive), 'turbo' (paralinguistic tags), or 'multilingual'",
            },
            "reference_audio": {
                "type": "str",
                "default": None,
                "help": "Path to reference audio file (~5-15s WAV/MP3) for zero-shot voice cloning",
            },
            "exaggeration": {
                "type": "float",
                "default": 0.0,
                "help": "Emotional expressiveness and dynamic range (0.0=neutral, 0.5=moderate, 1.0+=dramatic)",
            },
            "cfg_weight": {
                "type": "float",
                "default": 0.5,
                "help": "Classifier-free guidance weight for pacing and prompt adherence",
            },
            "temperature": {
                "type": "float",
                "default": 0.8,
                "help": "Sampling temperature (0.1 to 1.5)",
            },
            "top_p": {
                "type": "float",
                "default": 0.95,
                "help": "Nucleus sampling top-p probability threshold",
            },
            "language_id": {
                "type": "str",
                "default": "en",
                "help": "Language code for multilingual variant (e.g. en, es, fr, de, it, pt, ru, zh, ja)",
            },
            "device": {
                "type": "str",
                "default": "auto",
                "choices": ["auto", "cuda", "cpu", "mps"],
                "help": "Compute device ('cuda', 'cpu', 'mps', or 'auto')",
            },
        }

    def _resolve_device(self, requested_device: str) -> str:
        if requested_device != "auto":
            return requested_device
        try:
            import torch
            if torch.cuda.is_available():
                return "cuda"
            if hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
                return "mps"
        except ImportError:
            pass
        return "cpu"

    def _get_model(self, variant: str, device: str):
        key = f"{variant}_{device}"
        if key in self._cached_models:
            return self._cached_models[key]

        _ensure_chatterbox_imported()
        print(f"[Chatterbox] Loading {variant} model on {device} (first run will download weights)...")

        try:
            if variant == "turbo":
                from chatterbox.tts_turbo import ChatterboxTurboTTS
                model = ChatterboxTurboTTS.from_pretrained(device=device)
            elif variant == "multilingual":
                from chatterbox.mtl_tts import ChatterboxMultilingualTTS
                model = ChatterboxMultilingualTTS.from_pretrained(device=device)
            else:
                from chatterbox.tts import ChatterboxTTS
                model = ChatterboxTTS.from_pretrained(device=device)

            self._cached_models[key] = model
            return model
        except ImportError as e:
            raise RuntimeError(
                "Chatterbox library ('chatterbox-tts') is not installed.\n"
                "Install it with:\n"
                "  uv pip install chatterbox-tts"
            ) from e

    def synthesize(
        self,
        text: str,
        output_path: str,
        variant: str = "standard",
        reference_audio: Optional[str] = None,
        exaggeration: float = 0.0,
        cfg_weight: float = 0.5,
        temperature: float = 0.8,
        top_p: float = 0.95,
        language_id: str = "en",
        device: str = "auto",
        timestamps: bool = False,
        **kwargs: Any,
    ) -> Tuple[str, Optional[Any]]:
        """Synthesize speech using Chatterbox."""
        import re
        from ..timestamps import SentenceTimestamp, TimestampResult, interpolate_word_timestamps

        resolved_device = self._resolve_device(device)
        model = self._get_model(variant=variant, device=resolved_device)
        sr = getattr(model, "sr", 24000)

        # Reference audio prompt for cloning
        ref_path = reference_audio or kwargs.get("audio_prompt_path")
        abs_ref = os.path.abspath(ref_path) if ref_path and os.path.exists(ref_path) else None

        if timestamps:
            raw_sentences = [s.strip() for s in re.split(r"(?<=[.!?])\s+", text) if s.strip()]
            if not raw_sentences:
                raw_sentences = [text.strip()]

            audio_parts = []
            sentence_timings = []
            all_words = []
            current_time = 0.0

            for s in raw_sentences:
                gen_kwargs: Dict[str, Any] = {"text": s}
                if abs_ref:
                    gen_kwargs["audio_prompt_path"] = abs_ref
                if variant == "standard":
                    if exaggeration is not None:
                        gen_kwargs["exaggeration"] = float(exaggeration)
                    if cfg_weight is not None:
                        gen_kwargs["cfg_weight"] = float(cfg_weight)
                elif variant == "multilingual":
                    gen_kwargs["language_id"] = language_id
                    if cfg_weight is not None:
                        gen_kwargs["cfg_weight"] = float(cfg_weight)

                wav = model.generate(**gen_kwargs)
                if hasattr(wav, "detach"):
                    wav_np = wav.detach().cpu().numpy().squeeze()
                else:
                    wav_np = np.asarray(wav).squeeze()

                dur = len(wav_np) / float(sr)
                sent_start = current_time
                sent_end = current_time + dur
                words = interpolate_word_timestamps(s, sent_start, sent_end)

                sentence_timings.append(SentenceTimestamp(sentence=s, start=sent_start, end=sent_end, words=words))
                all_words.extend(words)
                audio_parts.append(wav_np)
                current_time = sent_end

            combined = np.concatenate(audio_parts) if audio_parts else np.zeros(2400, dtype=np.float32)
            ts_result = TimestampResult(sentences=sentence_timings, words=all_words, total_duration=current_time)
            saved_file = save_audio(combined, output_path, sample_rate=sr)
            return saved_file, ts_result

        # Fast single pass without timestamps
        gen_kwargs = {"text": text}
        if abs_ref:
            gen_kwargs["audio_prompt_path"] = abs_ref
        if variant == "standard":
            if exaggeration is not None:
                gen_kwargs["exaggeration"] = float(exaggeration)
            if cfg_weight is not None:
                gen_kwargs["cfg_weight"] = float(cfg_weight)
        elif variant == "multilingual":
            gen_kwargs["language_id"] = language_id
            if cfg_weight is not None:
                gen_kwargs["cfg_weight"] = float(cfg_weight)

        wav = model.generate(**gen_kwargs)
        saved_file = save_audio(wav, output_path, sample_rate=sr)
        return saved_file, None
