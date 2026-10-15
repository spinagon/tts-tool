"""Audio utilities for TTS CLI: saving, converting, and playing generated audio."""

import os
import shutil
import subprocess
import sys
from typing import Optional, Union
import numpy as np


def save_audio(
    audio_data: Union[np.ndarray, "torch.Tensor"],  # noqa: F821
    output_path: str,
    sample_rate: int = 24000,
) -> str:
    """Save raw audio data (numpy array or torch tensor) to a file (wav, mp3, ogg, flac).

    Args:
        audio_data: 1D or 2D audio samples (numpy array or torch tensor).
        output_path: Target file path.
        sample_rate: Audio sampling rate in Hz.

    Returns:
        The canonical path to the saved audio file.
    """
    # Convert PyTorch tensor to numpy if torch is loaded
    if hasattr(audio_data, "detach"):
        audio_data = audio_data.detach().cpu().numpy()

    # Ensure float32 numpy array
    if not isinstance(audio_data, np.ndarray):
        audio_data = np.asarray(audio_data, dtype=np.float32)

    # Flatten if multi-channel 1D or squeeze extra dimensions
    audio_data = np.squeeze(audio_data)

    # Clip to avoid clipping distortion
    if audio_data.dtype in (np.float32, np.float64):
        max_val = np.max(np.abs(audio_data))
        if max_val > 1.0:
            audio_data = audio_data / max_val

    output_dir = os.path.dirname(os.path.abspath(output_path))
    if output_dir:
        os.makedirs(output_dir, exist_ok=True)

    ext = os.path.splitext(output_path)[1].lower()
    temp_wav = output_path if ext == ".wav" else output_path + ".temp.wav"

    # Write WAV first using soundfile or wave module
    saved = False
    try:
        import soundfile as sf
        sf.write(temp_wav, audio_data, sample_rate)
        saved = True
    except ImportError:
        pass

    if not saved:
        try:
            import scipy.io.wavfile as wavfile
            # Convert float32 [-1, 1] to int16 for standard wavfile compatibility
            int16_data = np.clip(audio_data * 32767.0, -32768, 32767).astype(np.int16)
            wavfile.write(temp_wav, sample_rate, int16_data)
            saved = True
        except ImportError:
            pass

    if not saved:
        # Fallback to standard library wave module
        import struct
        import wave
        int16_data = np.clip(audio_data * 32767.0, -32768, 32767).astype(np.int16)
        with wave.open(temp_wav, "wb") as wf:
            wf.setnchannels(1)
            wf.setsampwidth(2)
            wf.setframerate(sample_rate)
            wf.writeframes(int16_data.tobytes())
        saved = True

    # Convert to desired extension using ffmpeg if not wav
    if ext and ext != ".wav":
        if shutil.which("ffmpeg"):
            cmd = ["ffmpeg", "-y", "-i", temp_wav, output_path]
            res = subprocess.run(cmd, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
            if res.returncode == 0:
                if os.path.exists(temp_wav) and temp_wav != output_path:
                    os.remove(temp_wav)
                return os.path.abspath(output_path)
            else:
                print(f"[Warning] ffmpeg conversion to {ext} failed; keeping {temp_wav}", file=sys.stderr)
                return os.path.abspath(temp_wav)
        else:
            print(f"[Warning] ffmpeg not found; cannot convert to {ext}. Saved as WAV: {temp_wav}", file=sys.stderr)
            return os.path.abspath(temp_wav)

    return os.path.abspath(output_path)


def play_audio(audio_path: str) -> bool:
    """Play an audio file using available system media players."""
    players = [
        ["ffplay", "-nodisp", "-autoexit"],
        ["mpv", "--no-video"],
        ["paplay"],
        ["aplay"],
        ["play"],  # sox
        ["xdg-open"],
    ]

    for player in players:
        binary = player[0]
        if shutil.which(binary):
            try:
                subprocess.run(player + [audio_path], check=True)
                return True
            except Exception:
                continue

    print(f"[Warning] No supported audio player found (tried ffplay, mpv, aplay, etc.). File saved at: {audio_path}", file=sys.stderr)
    return False
