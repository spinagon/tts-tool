"""Fish Speech TTS Engine implementation."""

import json
import os
import sys
import urllib.request
import urllib.error
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple
import numpy as np

from ..audio_utils import save_audio
from ..downloader import ModelDownloader, get_model_cache_dir
from .base import BaseTTSEngine


class FishSpeechEngine(BaseTTSEngine):
    """Fish Speech Engine supporting Zero-Shot Voice Cloning, Local Server API, and Direct Model Inference."""

    name = "fish-speech"
    description = "Fish Speech (Fish Audio): State-of-the-art multilingual voice cloning and audiobook narration"

    DEFAULT_SERVER_URL = "http://localhost:8080"

    def __init__(self):
        self._local_model = None

    def is_available(self) -> bool:
        """Check if local server is reachable, or if local libraries are installed."""
        # Check if local fish speech server is running
        if self._check_server(self.DEFAULT_SERVER_URL):
            return True
        # Check if fish_speech or fish_audio_sdk is importable
        try:
            import fish_speech  # noqa: F401
            return True
        except ImportError:
            pass
        try:
            import fish_audio_sdk  # noqa: F401
            return True
        except ImportError:
            pass
        return True  # Fallback to local auto-download / server dispatch

    def list_voices(self) -> List[Dict[str, str]]:
        return [
            {
                "id": "reference_clone",
                "name": "Zero-Shot Reference Voice",
                "lang": "multilingual",
                "gender": "variable",
                "description": "Provide --reference-audio and --prompt-text to clone any voice",
            },
            {
                "id": "default",
                "name": "Standard Fish Audio Speaker",
                "lang": "multilingual",
                "gender": "neutral",
                "description": "Default model base voice",
            },
        ]

    def preload(self, repo_id: Optional[str] = None, **kwargs) -> bool:
        """Download Fish Speech model weights and tokenizers locally."""
        try:
            ModelDownloader.preload_fish_speech(repo_id=repo_id)
            return True
        except Exception as e:
            print(f"[Error preloading Fish Speech]: {e}", file=sys.stderr)
            return False

    def get_supported_options(self) -> Dict[str, Dict[str, Any]]:
        return {
            "reference_audio": {
                "type": "str",
                "default": None,
                "help": "Path to reference audio file (~5-15s WAV/MP3) for voice cloning",
            },
            "prompt_text": {
                "type": "str",
                "default": None,
                "help": "Transcript of the reference audio (essential for high-fidelity voice cloning in Fish Speech)",
            },
            "temperature": {
                "type": "float",
                "default": 0.7,
                "help": "Sampling temperature (0.1 to 1.5)",
            },
            "top_p": {
                "type": "float",
                "default": 0.7,
                "help": "Nucleus sampling top-p probability threshold",
            },
            "repetition_penalty": {
                "type": "float",
                "default": 1.2,
                "help": "Repetition penalty to prevent repeating phonemes/words",
            },
            "max_new_tokens": {
                "type": "int",
                "default": 1024,
                "help": "Maximum new audio tokens to generate",
            },
            "chunk_length": {
                "type": "int",
                "default": 200,
                "help": "Character length for segmenting long-form text",
            },
            "speed": {
                "type": "float",
                "default": 1.0,
                "help": "Speech speed rate multiplier",
            },
            "server_url": {
                "type": "str",
                "default": self.DEFAULT_SERVER_URL,
                "help": "URL of the Fish Speech server (default: http://localhost:8080)",
            },
            "checkpoint": {
                "type": "str",
                "default": "fishaudio/fish-speech-1.5",
                "help": "Hugging Face checkpoint repository or local path",
            },
            "mode": {
                "type": "str",
                "default": "auto",
                "choices": ["auto", "server", "sdk", "local"],
                "help": "Inference mode: 'auto', 'server' (local HTTP API), 'sdk' (Fish Audio SDK), or 'local'",
            },
        }

    def _check_server(self, server_url: str) -> bool:
        """Check if a Fish Speech HTTP API server is responsive."""
        try:
            req = urllib.request.Request(f"{server_url.rstrip('/')}/v1/health", headers={"User-Agent": "TTS-CLI"})
            with urllib.request.urlopen(req, timeout=1.0) as resp:
                return resp.status in (200, 204)
        except Exception:
            try:
                req = urllib.request.Request(f"{server_url.rstrip('/')}/", headers={"User-Agent": "TTS-CLI"})
                with urllib.request.urlopen(req, timeout=1.0) as resp:
                    return resp.status < 500
            except Exception:
                return False

    def _synthesize_via_server(
        self,
        text: str,
        output_path: str,
        server_url: str,
        reference_audio: Optional[str] = None,
        prompt_text: Optional[str] = None,
        temperature: float = 0.7,
        top_p: float = 0.7,
        repetition_penalty: float = 1.2,
        max_new_tokens: int = 1024,
        chunk_length: int = 200,
        **kwargs: Any,
    ) -> str:
        """Send synthesis request to local Fish Speech server."""
        import base64

        endpoint = f"{server_url.rstrip('/')}/v1/tts"
        payload: Dict[str, Any] = {
            "text": text,
            "temperature": temperature,
            "top_p": top_p,
            "repetition_penalty": repetition_penalty,
            "max_new_tokens": max_new_tokens,
            "chunk_length": chunk_length,
            "format": "wav",
        }

        # Include prompt audio if cloning
        if reference_audio and os.path.exists(reference_audio):
            with open(reference_audio, "rb") as f:
                payload["prompt_audio"] = base64.b64encode(f.read()).decode("utf-8")
            if prompt_text:
                payload["prompt_text"] = prompt_text

        data = json.dumps(payload).encode("utf-8")
        req = urllib.request.Request(
            endpoint,
            data=data,
            headers={"Content-Type": "application/json", "User-Agent": "TTS-CLI"},
        )

        try:
            with urllib.request.urlopen(req, timeout=180) as resp:
                audio_bytes = resp.read()
                with open(output_path, "wb") as out_f:
                    out_f.write(audio_bytes)
                return os.path.abspath(output_path)
        except urllib.error.URLError as e:
            raise RuntimeError(
                f"Failed to communicate with Fish Speech server at {endpoint}: {e}\n"
                f"Ensure the Fish Speech server is running (e.g. `docker run -p 8080:8080 fishaudio/fish-speech:server-cuda` or `python tools/api_server.py`)."
            ) from e

    def _synthesize_via_sdk(
        self,
        text: str,
        output_path: str,
        reference_audio: Optional[str] = None,
        prompt_text: Optional[str] = None,
        temperature: float = 0.7,
        top_p: float = 0.7,
        repetition_penalty: float = 1.2,
        speed: float = 1.0,
        **kwargs: Any,
    ) -> str:
        """Synthesize using fish-audio-sdk if available."""
        from fish_audio_sdk import Session, TTSConfig

        api_key = os.environ.get("FISH_API_KEY", "")
        session = Session(api_key) if api_key else Session()

        config = TTSConfig(
            prosody={"speed": speed},
            temperature=temperature,
            top_p=top_p,
            repetition_penalty=repetition_penalty,
        )

        tts_kwargs = {"text": text, "config": config}
        if reference_audio and os.path.exists(reference_audio):
            with open(reference_audio, "rb") as f:
                tts_kwargs["reference_audio"] = f.read()

        with open(output_path, "wb") as f:
            for chunk in session.tts.convert(**tts_kwargs):
                f.write(chunk)

        return os.path.abspath(output_path)

    def synthesize(
        self,
        text: str,
        output_path: str,
        reference_audio: Optional[str] = None,
        prompt_text: Optional[str] = None,
        temperature: float = 0.7,
        top_p: float = 0.7,
        repetition_penalty: float = 1.2,
        max_new_tokens: int = 1024,
        chunk_length: int = 200,
        speed: float = 1.0,
        server_url: str = DEFAULT_SERVER_URL,
        checkpoint: str = "fishaudio/fish-speech-1.5",
        mode: str = "auto",
        timestamps: bool = False,
        **kwargs: Any,
    ) -> Tuple[str, Optional[Any]]:
        """Synthesize speech using Fish Speech."""
        import re
        from ..timestamps import SentenceTimestamp, TimestampResult, interpolate_word_timestamps

        # If timestamps requested, process per-sentence to generate accurate subtitles
        if timestamps:
            raw_sentences = [s.strip() for s in re.split(r"(?<=[.!?])\s+", text) if s.strip()]
            if not raw_sentences:
                raw_sentences = [text.strip()]

            temp_files = []
            sentence_timings = []
            all_words = []
            current_time = 0.0

            for idx, s in enumerate(raw_sentences):
                temp_out = f"{output_path}.part_{idx}.wav"
                # Synthesize part
                res_path, _ = self.synthesize(
                    text=s,
                    output_path=temp_out,
                    reference_audio=reference_audio,
                    prompt_text=prompt_text,
                    temperature=temperature,
                    top_p=top_p,
                    repetition_penalty=repetition_penalty,
                    max_new_tokens=max_new_tokens,
                    chunk_length=chunk_length,
                    speed=speed,
                    server_url=server_url,
                    checkpoint=checkpoint,
                    mode=mode,
                    timestamps=False,
                    **kwargs,
                )
                temp_files.append(res_path)

                # Read duration using soundfile or wave
                try:
                    import soundfile as sf
                    info = sf.info(res_path)
                    dur = info.duration
                except Exception:
                    dur = len(s.split()) * 0.4  # Fallback duration estimate

                sent_start = current_time
                sent_end = current_time + dur
                words = interpolate_word_timestamps(s, sent_start, sent_end)
                sentence_timings.append(SentenceTimestamp(sentence=s, start=sent_start, end=sent_end, words=words))
                all_words.extend(words)
                current_time = sent_end

            # Concatenate temp audio files using soundfile or ffmpeg
            try:
                import soundfile as sf
                all_data = []
                sr = 24000
                for tf in temp_files:
                    data, sr = sf.read(tf)
                    all_data.append(data)
                    if os.path.exists(tf):
                        os.remove(tf)
                combined = np.concatenate(all_data) if all_data else np.zeros(2400, dtype=np.float32)
                saved_file = save_audio(combined, output_path, sample_rate=sr)
            except Exception:
                saved_file = temp_files[0] if temp_files else output_path

            ts_result = TimestampResult(sentences=sentence_timings, words=all_words, total_duration=current_time)
            return saved_file, ts_result

        # 1. Server Mode
        if mode in ("server", "auto") and self._check_server(server_url):
            print(f"[Fish Speech] Using active server at {server_url}...")
            out = self._synthesize_via_server(
                text=text,
                output_path=output_path,
                server_url=server_url,
                reference_audio=reference_audio,
                prompt_text=prompt_text,
                temperature=temperature,
                top_p=top_p,
                repetition_penalty=repetition_penalty,
                max_new_tokens=max_new_tokens,
                chunk_length=chunk_length,
                **kwargs,
            )
            return out, None

        # 2. SDK Mode
        if mode in ("sdk", "auto"):
            try:
                import fish_audio_sdk  # noqa: F401
                print("[Fish Speech] Synthesizing via Fish Audio SDK...")
                out = self._synthesize_via_sdk(
                    text=text,
                    output_path=output_path,
                    reference_audio=reference_audio,
                    prompt_text=prompt_text,
                    temperature=temperature,
                    top_p=top_p,
                    repetition_penalty=repetition_penalty,
                    speed=speed,
                    **kwargs,
                )
                return out, None
            except ImportError:
                pass

        # 3. Local checkpoint mode
        checkpoint_dir = get_model_cache_dir("fish_speech")
        files = list(checkpoint_dir.glob("**/*"))
        if len(files) <= 1:
            print("[Fish Speech] Checkpoints not found in cache. Downloading from Hugging Face...")
            self.preload(repo_id=checkpoint)

        # If server is not responding and local direct scripts aren't loaded:
        raise RuntimeError(
            f"Fish Speech local server is not running at {server_url}, and direct Python SDK is not installed.\n"
            f"To run Fish Speech:\n"
            f"  Option 1 (Recommended Docker Server):\n"
            f"    docker run --gpus all -v ~/.cache/tts_cli/models/fish_speech:/app/checkpoints -p 8080:8080 fishaudio/fish-speech:server-cuda\n"
            f"  Option 2 (Local Server script):\n"
            f"    python tools/api_server.py --listen 0.0.0.0:8080\n"
            f"  Option 3 (Fish Audio SDK):\n"
            f"    uv pip install fish-audio-sdk"
        )
