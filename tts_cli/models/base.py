"""Base class definition for all TTS model engines."""

from abc import ABC, abstractmethod
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, TYPE_CHECKING

if TYPE_CHECKING:
    from ..timestamps import TimestampResult


class BaseTTSEngine(ABC):
    """Abstract base class for a Text-to-Speech engine."""

    name: str = "base"
    description: str = "Base TTS Engine"

    @abstractmethod
    def is_available(self) -> bool:
        """Check if engine requirements or packages are installed and ready."""
        pass

    @abstractmethod
    def list_voices(self) -> List[Dict[str, str]]:
        """Return a list of available preset voices with details."""
        pass

    @abstractmethod
    def preload(self, **kwargs) -> bool:
        """Download and cache required weights and assets locally."""
        pass

    @abstractmethod
    def synthesize(
        self,
        text: str,
        output_path: str,
        timestamps: bool = False,
        **kwargs: Any,
    ) -> Tuple[str, Optional[Any]]:
        """Convert input text to speech and save to output_path.

        Args:
            text: Input text to convert.
            output_path: Target audio file path.
            timestamps: Whether to generate word/sentence alignment timestamps.

        Returns:
            Tuple of (audio_file_path, TimestampResult or None).
        """
        pass

    @abstractmethod
    def get_supported_options(self) -> Dict[str, Dict[str, Any]]:
        """Return a dictionary of options supported by this engine with their defaults and types."""
        pass
