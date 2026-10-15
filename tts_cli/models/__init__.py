"""Model engines registry for Kokoro, Chatterbox, and Fish Speech."""

from typing import Dict, Type
from .base import BaseTTSEngine
from .kokoro_engine import KokoroEngine
from .chatterbox_engine import ChatterboxEngine
from .fish_speech_engine import FishSpeechEngine


ENGINE_REGISTRY: Dict[str, Type[BaseTTSEngine]] = {
    "kokoro": KokoroEngine,
    "chatterbox": ChatterboxEngine,
    "fish-speech": FishSpeechEngine,
    "fish_speech": FishSpeechEngine,
    "fish": FishSpeechEngine,
}


def get_engine(name: str) -> BaseTTSEngine:
    """Retrieve an initialized TTS engine instance by name."""
    clean_name = name.lower().strip()
    if clean_name not in ENGINE_REGISTRY:
        supported = ", ".join(list_supported_engines())
        raise ValueError(f"Unknown TTS engine '{name}'. Supported engines: {supported}")
    engine_class = ENGINE_REGISTRY[clean_name]
    return engine_class()


def list_supported_engines() -> list:
    """List distinct canonical engine names."""
    return ["kokoro", "chatterbox", "fish-speech"]
