"""Centralized model downloader and cache manager for Kokoro, Chatterbox, and Fish Speech."""

import os
import sys
import urllib.request
from pathlib import Path
from typing import Callable, Dict, List, Optional, Tuple


def get_cache_root() -> Path:
    """Return the root directory for TTS model caches."""
    env_dir = os.environ.get("TTS_CACHE_DIR")
    if env_dir:
        path = Path(env_dir)
    else:
        path = Path.home() / ".cache" / "tts_cli"
    path.mkdir(parents=True, exist_ok=True)
    return path


def get_model_cache_dir(engine: str) -> Path:
    """Return model directory for a specific engine."""
    engine_dir = get_cache_root() / "models" / engine.lower()
    engine_dir.mkdir(parents=True, exist_ok=True)
    return engine_dir


def format_size(bytes_num: int) -> str:
    """Format bytes into human-readable string."""
    for unit in ["B", "KB", "MB", "GB", "TB"]:
        if abs(bytes_num) < 1024.0:
            return f"{bytes_num:3.1f} {unit}"
        bytes_num /= 1024.0
    return f"{bytes_num:.1f} PB"


def download_file_with_progress(url: str, dest_path: Path, desc: str = "") -> Path:
    """Download a file with terminal progress reporting."""
    dest_path.parent.mkdir(parents=True, exist_ok=True)
    temp_path = dest_path.with_suffix(dest_path.suffix + ".downloading")

    print(f"[Download] {desc or dest_path.name} from {url}")

    def report_progress(block_num, block_size, total_size):
        downloaded = block_num * block_size
        if total_size > 0:
            percent = min(100.0, downloaded * 100.0 / total_size)
            prog_bar = "=" * int(percent // 4) + ">" + " " * (25 - int(percent // 4))
            sys.stdout.write(
                f"\r  [{prog_bar}] {percent:5.1f}% ({format_size(downloaded)} / {format_size(total_size)})"
            )
        else:
            sys.stdout.write(f"\r  Downloaded: {format_size(downloaded)}")
        sys.stdout.flush()

    try:
        urllib.request.urlretrieve(url, temp_path, reporthook=report_progress)
        sys.stdout.write("\n")
        temp_path.rename(dest_path)
        print(f"[Success] Saved {dest_path.name} ({format_size(dest_path.stat().st_size)})")
        return dest_path
    except Exception as e:
        sys.stdout.write("\n")
        if temp_path.exists():
            temp_path.unlink()
        raise RuntimeError(f"Failed to download {url}: {e}") from e


def hf_download_file(
    repo_id: str,
    filename: str,
    subfolder: Optional[str] = None,
    local_dir: Optional[Path] = None,
) -> Path:
    """Download a single file from Hugging Face Hub using huggingface_hub or direct URL fallback."""
    # Try huggingface_hub library first if installed
    try:
        from huggingface_hub import hf_hub_download
        downloaded = hf_hub_download(
            repo_id=repo_id,
            filename=filename,
            subfolder=subfolder,
            local_dir=str(local_dir) if local_dir else None,
            local_dir_use_symlinks=False,
        )
        return Path(downloaded)
    except (ImportError, Exception):
        pass

    # Direct fallback download via Hugging Face CDN URL
    if subfolder:
        remote_path = f"{subfolder}/{filename}"
    else:
        remote_path = filename

    url = f"https://huggingface.co/{repo_id}/resolve/main/{remote_path}"
    target_dir = local_dir if local_dir else (get_cache_root() / "hf_fallback" / repo_id.replace("/", "_"))
    target_path = target_dir / filename
    if target_path.exists() and target_path.stat().st_size > 0:
        return target_path

    return download_file_with_progress(url, target_path, desc=f"{repo_id}/{filename}")


class ModelDownloader:
    """Manages preloading and availability checks for all supported TTS models."""

    # 1. KOKORO
    KOKORO_ONNX_REPO = "onnx-community/Kokoro-82M-ONNX"
    KOKORO_ONNX_MODEL = "onnx/model.onnx"
    KOKORO_ONNX_VOICES = "voices.bin"
    # Alternative direct Kokoro assets
    KOKORO_FASTRTC_REPO = "fastrtc/kokoro-onnx"
    KOKORO_PT_REPO = "hexgrad/Kokoro-82M"

    # 2. CHATTERBOX
    CHATTERBOX_REPO = "resemble-ai/chatterbox"

    # 3. FISH SPEECH
    FISH_SPEECH_REPO = "fishaudio/fish-speech-1.5"

    @classmethod
    def preload_kokoro(cls, onnx: bool = True) -> Dict[str, Path]:
        """Preload Kokoro weights and voice binaries."""
        kokoro_dir = get_model_cache_dir("kokoro")
        print("\n=== Preloading Kokoro TTS Assets ===")

        # We download both ONNX model and voice bins for instant, zero-dependency CPU inference
        onnx_file = kokoro_dir / "kokoro-v1.0.onnx"
        voices_file = kokoro_dir / "voices-v1.0.bin"

        results = {}
        if not onnx_file.exists():
            try:
                # Try fastrtc / hexgrad assets
                p = hf_download_file(cls.KOKORO_FASTRTC_REPO, "kokoro-v1.0.onnx", local_dir=kokoro_dir)
                results["model"] = p
            except Exception:
                p = hf_download_file(cls.KOKORO_ONNX_REPO, "onnx/model.onnx", local_dir=kokoro_dir)
                results["model"] = p
        else:
            print(f"[Cached] Kokoro ONNX model: {onnx_file} ({format_size(onnx_file.stat().st_size)})")
            results["model"] = onnx_file

        if not voices_file.exists():
            try:
                p = hf_download_file(cls.KOKORO_FASTRTC_REPO, "voices-v1.0.bin", local_dir=kokoro_dir)
                results["voices"] = p
            except Exception:
                p = hf_download_file(cls.KOKORO_ONNX_REPO, "voices.bin", local_dir=kokoro_dir)
                results["voices"] = p
        else:
            print(f"[Cached] Kokoro voices: {voices_file} ({format_size(voices_file.stat().st_size)})")
            results["voices"] = voices_file

        print("[Kokoro] Preload complete!")
        return results

    @classmethod
    def preload_kokoro_timestamped(cls) -> Path:
        """Preload the timestamped Kokoro ONNX model exposing duration tensors."""
        kokoro_dir = get_model_cache_dir("kokoro")
        ts_model_path = kokoro_dir / "kokoro-v1.0-timestamped.onnx"
        if ts_model_path.exists() and ts_model_path.stat().st_size > 0:
            return ts_model_path

        print("\n[Kokoro] Downloading timestamped ONNX model for word/sentence timing...")
        try:
            from huggingface_hub import hf_hub_download
            downloaded = hf_hub_download(
                repo_id="onnx-community/Kokoro-82M-v1.0-ONNX-timestamped",
                filename="onnx/model_quantized.onnx",
            )
            import shutil
            shutil.copy2(downloaded, str(ts_model_path))
            return ts_model_path
        except Exception:
            return hf_download_file(
                "onnx-community/Kokoro-82M-v1.0-ONNX-timestamped",
                "onnx/model_quantized.onnx",
                local_dir=kokoro_dir,
            )

    @classmethod
    def preload_chatterbox(cls, variant: str = "standard") -> Path:
        """Preload Chatterbox model weights from Hugging Face."""
        cache_dir = get_model_cache_dir("chatterbox")
        print(f"\n=== Preloading Chatterbox TTS ({variant}) ===")

        # Attempt to preload using chatterbox library if installed
        try:
            from chatterbox.tts import ChatterboxTTS
            from chatterbox.tts_turbo import ChatterboxTurboTTS
            from chatterbox.mtl_tts import ChatterboxMultilingualTTS

            if variant == "turbo":
                print("[Chatterbox] Loading ChatterboxTurboTTS into cache...")
                _ = ChatterboxTurboTTS.from_pretrained(device="cpu")
            elif variant == "multilingual":
                print("[Chatterbox] Loading ChatterboxMultilingualTTS into cache...")
                _ = ChatterboxMultilingualTTS.from_pretrained(device="cpu")
            else:
                print("[Chatterbox] Loading standard ChatterboxTTS into cache...")
                _ = ChatterboxTTS.from_pretrained(device="cpu")

            print("[Chatterbox] Preload complete via chatterbox package!")
            return cache_dir
        except (ImportError, Exception) as e:
            print(f"[Info] chatterbox package auto-load note: {e}")

        # Alternatively download via huggingface_hub snapshot_download
        try:
            from huggingface_hub import snapshot_download
            print(f"[Chatterbox] Downloading snapshot from {cls.CHATTERBOX_REPO}...")
            local_path = snapshot_download(
                repo_id=cls.CHATTERBOX_REPO,
                local_dir=str(cache_dir),
                local_dir_use_symlinks=False,
            )
            print(f"[Chatterbox] Downloaded to {local_path}")
            return Path(local_path)
        except Exception as e:
            print(f"[Warning] Direct snapshot download note: {e}")
            return cache_dir

    @classmethod
    def preload_fish_speech(cls, repo_id: Optional[str] = None) -> Path:
        """Preload Fish Speech checkpoint and codec weights."""
        repo = repo_id or cls.FISH_SPEECH_REPO
        cache_dir = get_model_cache_dir("fish_speech")
        print(f"\n=== Preloading Fish Speech ({repo}) ===")

        try:
            from huggingface_hub import snapshot_download
            print(f"[Fish Speech] Downloading repository snapshot for {repo}...")
            local_path = snapshot_download(
                repo_id=repo,
                local_dir=str(cache_dir),
                local_dir_use_symlinks=False,
            )
            print(f"[Fish Speech] Checkpoints downloaded to: {local_path}")
            return Path(local_path)
        except Exception as e:
            print(f"[Warning] Could not snapshot download Fish Speech ({e}).")
            print("Note: If using Fish Speech via local API server (e.g. docker or tools/api_server.py), checkpoints are loaded on the server.")
            return cache_dir

    @classmethod
    def preload_all(cls) -> Dict[str, bool]:
        """Preload all three TTS engines."""
        results = {}
        for engine, func in [
            ("kokoro", cls.preload_kokoro),
            ("chatterbox", cls.preload_chatterbox),
            ("fish-speech", cls.preload_fish_speech),
        ]:
            try:
                func()
                results[engine] = True
            except Exception as e:
                print(f"[Error preloading {engine}]: {e}", file=sys.stderr)
                results[engine] = False
        return results

    @classmethod
    def get_status(cls) -> Dict[str, Dict[str, str]]:
        """Return download / cache status of all models."""
        status = {}

        # Kokoro status
        k_dir = get_model_cache_dir("kokoro")
        k_model = k_dir / "kokoro-v1.0.onnx"
        k_voices = k_dir / "voices-v1.0.bin"
        if k_model.exists() and k_voices.exists():
            k_size = format_size(k_model.stat().st_size + k_voices.stat().st_size)
            status["kokoro"] = {"status": "Ready", "path": str(k_dir), "size": k_size}
        else:
            status["kokoro"] = {"status": "Not downloaded (will auto-download at first use)", "path": str(k_dir), "size": "0 B"}

        # Chatterbox status
        c_dir = get_model_cache_dir("chatterbox")
        c_files = list(c_dir.glob("**/*"))
        if len(c_files) > 1:
            total_size = sum(f.stat().st_size for f in c_files if f.is_file())
            status["chatterbox"] = {"status": "Ready", "path": str(c_dir), "size": format_size(total_size)}
        else:
            status["chatterbox"] = {"status": "Available via HF hub cache / chatterbox package", "path": str(c_dir), "size": "-"}

        # Fish speech status
        f_dir = get_model_cache_dir("fish_speech")
        f_files = list(f_dir.glob("**/*"))
        if len(f_files) > 1:
            total_size = sum(f.stat().st_size for f in f_files if f.is_file())
            status["fish-speech"] = {"status": "Ready", "path": str(f_dir), "size": format_size(total_size)}
        else:
            status["fish-speech"] = {"status": "Available / Server-ready (or auto-download on first use)", "path": str(f_dir), "size": "-"}

        return status
