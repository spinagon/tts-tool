"""Main Command Line Interface for TTS Tool."""

import argparse
import os
import sys
from pathlib import Path
from typing import Optional

from .audio_utils import play_audio
from .downloader import ModelDownloader, get_cache_root
from .models import get_engine, list_supported_engines


def read_input_text(text_arg: Optional[str], file_arg: Optional[str]) -> str:
    """Read input text from command-line argument, input file, or piped stdin."""
    if text_arg and text_arg.strip():
        return text_arg.strip()

    if file_arg:
        path = Path(file_arg)
        if not path.is_file():
            print(f"[Error] Input file not found: {file_arg}", file=sys.stderr)
            sys.exit(1)
        return path.read_text(encoding="utf-8").strip()

    # Check if data is piped via standard input
    if not sys.stdin.isatty():
        content = sys.stdin.read().strip()
        if content:
            return content

    return ""


def handle_preload(args: argparse.Namespace) -> None:
    """Handle model preloading command."""
    engine = getattr(args, "engine", "all").lower()
    print(f"Preloading TTS models (Target: {engine})...")
    print(f"Cache location: {get_cache_root()}")

    if engine == "all":
        ModelDownloader.preload_all()
    elif engine == "kokoro":
        ModelDownloader.preload_kokoro()
    elif engine == "chatterbox":
        variant = getattr(args, "variant", "standard")
        ModelDownloader.preload_chatterbox(variant=variant)
    elif engine in ("fish-speech", "fish"):
        repo = getattr(args, "checkpoint", None)
        ModelDownloader.preload_fish_speech(repo_id=repo)
    else:
        print(f"[Error] Unknown engine '{engine}'. Choose from: all, kokoro, chatterbox, fish-speech", file=sys.stderr)
        sys.exit(1)

    print("\nPreload process completed successfully!")


def handle_status(args: argparse.Namespace) -> None:
    """Show status of downloaded models."""
    status = ModelDownloader.get_status()
    print("=" * 60)
    print("  Local TTS Models Cache Status")
    print(f"  Cache Directory: {get_cache_root()}")
    print("=" * 60)
    for name, info in status.items():
        print(f"\n[{name.upper()}]")
        print(f"  Status: {info['status']}")
        print(f"  Path:   {info['path']}")
        print(f"  Size:   {info['size']}")
    print("\n" + "=" * 60)


def handle_list_voices(args: argparse.Namespace) -> None:
    """List voices for the specified or all engines."""
    target_engine = getattr(args, "engine", None)
    engines_to_query = [target_engine] if target_engine and target_engine != "all" else list_supported_engines()

    for eng_name in engines_to_query:
        try:
            eng = get_engine(eng_name)
            voices = eng.list_voices()
            print(f"\n=== Voices for Engine: {eng_name} ({eng.description}) ===")
            print(f"{'ID':<18} {'Name':<22} {'Lang':<10} {'Gender/Type':<12} {'Description'}")
            print("-" * 80)
            for v in voices:
                v_id = v.get("id", "-")
                v_name = v.get("name", "-")
                v_lang = v.get("lang", "-")
                v_gender = v.get("gender", "-")
                v_desc = v.get("description", "")
                print(f"{v_id:<18} {v_name:<22} {v_lang:<10} {v_gender:<12} {v_desc}")
        except Exception as e:
            print(f"Could not query voices for {eng_name}: {e}", file=sys.stderr)


def run_synthesis(
    engine_name: str,
    text: str,
    output_path: str,
    play: bool,
    timestamps: Optional[str] = None,
    timestamps_level: str = "both",
    timestamps_file: Optional[str] = None,
    **kwargs,
) -> str:
    """Execute text-to-speech synthesis using the selected engine."""
    engine = get_engine(engine_name)
    print(f"[{engine_name.upper()}] Generating speech...")
    print(f"  Input: {text[:60]}..." if len(text) > 60 else f"  Input: {text}")
    print(f"  Output: {output_path}")

    wants_timestamps = bool(timestamps)
    res = engine.synthesize(text=text, output_path=output_path, timestamps=wants_timestamps, **kwargs)
    if isinstance(res, tuple):
        out, ts_result = res
    else:
        out, ts_result = res, None

    print(f"[Done] Audio generated successfully: {out}")

    if ts_result and timestamps:
        target_base = timestamps_file if timestamps_file else out
        saved_files = ts_result.save(target_base, fmt=timestamps, level=timestamps_level)
        print("[Timestamps] Synchronized timing files generated:")
        for sf in saved_files:
            print(f"  -> {sf}")

    if play:
        print("[Play] Playing audio...")
        play_audio(out)

    return out


def build_parser() -> argparse.ArgumentParser:
    """Construct argument parser with subcommands and extensive options."""
    parser = argparse.ArgumentParser(
        prog="tts",
        description="Unified Text-to-Speech CLI supporting Kokoro-82M, Chatterbox, and Fish Speech.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""Examples:
  # Quick generation with Kokoro (default voice: af_heart)
  tts kokoro "Hello world, this is Kokoro running locally." -o hello.wav --play

  # Generate audio with synced word- and sentence-level subtitles (JSON, SRT, VTT, or CSV)
  tts kokoro "Welcome to automated video generation. Subtitles are now perfectly synced!" --timestamps srt -o video.wav

  # Custom voice and speed with Kokoro
  tts kokoro "Pipelines are running smoothly." --voice bm_george --speed 1.1 -o george.wav

  # Zero-shot voice cloning with Chatterbox
  tts chatterbox "Welcome back to the studio." --reference-audio sample.wav --exaggeration 0.7 -o cloned.wav

  # Emotional speech with paralinguistic tag in Chatterbox Turbo
  tts chatterbox "Well [chuckle], that was completely unexpected!" --variant turbo -o funny.wav

  # Voice cloning with Fish Speech
  tts fish-speech "This is Fish Speech in action." --reference-audio sample.wav --prompt-text "Sample transcript" -o fish.wav

  # Preload models ahead of time
  tts preload --engine all
  tts preload --engine kokoro

  # Pipe text from stdin
  echo "Reading from standard input pipe." | tts kokoro -o pipe.wav
""",
    )

    subparsers = parser.add_subparsers(dest="subcommand", help="Available subcommands")

    # Common parent parser for synthesis commands
    synth_parent = argparse.ArgumentParser(add_help=False)
    synth_parent.add_argument("text", nargs="?", default="", help="Text to convert to speech")
    synth_parent.add_argument("-t", "--text-arg", dest="text_flag", help="Text to convert to speech (flag alternative)")
    synth_parent.add_argument("-f", "--file", help="Path to text file to read")
    synth_parent.add_argument("-o", "--output", default="output.wav", help="Output audio file path (.wav, .mp3, .ogg, .flac). Default: output.wav")
    synth_parent.add_argument("-p", "--play", action="store_true", help="Automatically play audio after generation")
    synth_parent.add_argument(
        "--timestamps",
        nargs="?",
        const="json",
        choices=["json", "srt", "vtt", "csv", "all"],
        help="Generate synchronized subtitle/graphic timestamps (formats: json, srt, vtt, csv, all. Default: json)",
    )
    synth_parent.add_argument(
        "--timestamps-level",
        choices=["sentence", "word", "both"],
        default="both",
        help="Granularity level for subtitles/timestamps (default: both)",
    )
    synth_parent.add_argument("--timestamps-file", help="Custom output path base for timestamp/subtitle files")

    # 1. KOKORO SUBCOMMAND
    p_kokoro = subparsers.add_parser("kokoro", parents=[synth_parent], help="Synthesize text using Kokoro-82M (Fast CPU/GPU)")
    p_kokoro.add_argument("-v", "--voice", default="af_heart", help="Preset voice ID (default: af_heart). Run 'tts list-voices --engine kokoro' for full list.")
    p_kokoro.add_argument("-s", "--speed", type=float, default=1.0, help="Speech speed multiplier (default: 1.0, range: 0.5-2.0)")
    p_kokoro.add_argument("-l", "--lang", dest="lang_code", default="auto", help="Language code ('a', 'b', 'j', 'z', 'e', 'f', 'h', 'i', 'p', or 'auto')")
    p_kokoro.add_argument("--backend", choices=["onnx", "pytorch"], default="onnx", help="Inference backend (default: onnx)")
    p_kokoro.add_argument("--sentence-pause", type=float, default=0.25, help="Pause duration in seconds between sentences (ONNX backend, default: 0.25)")
    p_kokoro.add_argument("--clause-pause", type=float, default=0.1, help="Pause duration in seconds between clauses (ONNX backend, default: 0.1)")
    p_kokoro.add_argument("--continuous", action="store_true", help="Synthesize with continuous overlapping windows for smoother prosody (ONNX)")
    p_kokoro.add_argument("--no-trim", dest="trim", action="store_false", default=True, help="Do not trim silence from start and end")
    p_kokoro.add_argument("--split-pattern", default=r"\n+", help="Regex pattern for text segment chunking")

    # 2. CHATTERBOX SUBCOMMAND
    p_chatterbox = subparsers.add_parser("chatterbox", parents=[synth_parent], help="Synthesize text using Chatterbox (Voice Cloning & Expressiveness)")
    p_chatterbox.add_argument("--variant", choices=["standard", "turbo", "multilingual"], default="standard", help="Chatterbox variant (default: standard)")
    p_chatterbox.add_argument("-r", "--reference-audio", help="Reference audio file (~5-15s WAV/MP3) for zero-shot voice cloning")
    p_chatterbox.add_argument("-e", "--exaggeration", type=float, default=0.0, help="Emotional intensity / expressiveness (0.0 to 1.5, default: 0.0)")
    p_chatterbox.add_argument("--cfg-weight", type=float, default=0.5, help="Classifier-Free Guidance weight for pacing (default: 0.5)")
    p_chatterbox.add_argument("--temperature", type=float, default=0.8, help="Sampling temperature (default: 0.8)")
    p_chatterbox.add_argument("--top-p", type=float, default=0.95, help="Nucleus sampling top-p (default: 0.95)")
    p_chatterbox.add_argument("--language-id", default="en", help="Language ID for multilingual variant (e.g. en, es, fr, de, it, ru, zh, ja)")
    p_chatterbox.add_argument("--device", choices=["auto", "cuda", "cpu", "mps"], default="auto", help="Compute device (default: auto)")

    # 3. FISH SPEECH SUBCOMMAND
    p_fish = subparsers.add_parser("fish-speech", parents=[synth_parent], aliases=["fish"], help="Synthesize text using Fish Speech (Long-form & Cloning)")
    p_fish.add_argument("-r", "--reference-audio", help="Reference audio file (~5-15s WAV/MP3) for voice cloning")
    p_fish.add_argument("--prompt-text", help="Transcript of reference audio (essential for Fish Speech voice cloning)")
    p_fish.add_argument("--temperature", type=float, default=0.7, help="Sampling temperature (default: 0.7)")
    p_fish.add_argument("--top-p", type=float, default=0.7, help="Nucleus sampling top-p (default: 0.7)")
    p_fish.add_argument("--repetition-penalty", type=float, default=1.2, help="Repetition penalty (default: 1.2)")
    p_fish.add_argument("--max-new-tokens", type=int, default=1024, help="Maximum new audio tokens (default: 1024)")
    p_fish.add_argument("--chunk-length", type=int, default=200, help="Character chunk length for long-form synthesis (default: 200)")
    p_fish.add_argument("-s", "--speed", type=float, default=1.0, help="Speed rate multiplier (default: 1.0)")
    p_fish.add_argument("--server-url", default="http://localhost:8080", help="URL of the local/remote Fish Speech API server (default: http://localhost:8080)")
    p_fish.add_argument("--checkpoint", default="fishaudio/fish-speech-1.5", help="Model checkpoint repository or path (default: fishaudio/fish-speech-1.5)")
    p_fish.add_argument("--mode", choices=["auto", "server", "sdk", "local"], default="auto", help="Execution mode (default: auto)")

    # 4. GENERAL SPEAK COMMAND
    p_speak = subparsers.add_parser("speak", parents=[synth_parent], help="Synthesize text with explicit engine selector")
    p_speak.add_argument("-e", "--engine", choices=list_supported_engines(), default="kokoro", help="TTS Engine to use (default: kokoro)")
    # Forward engine-specific options through kwargs
    p_speak.add_argument("-v", "--voice", default="af_heart")
    p_speak.add_argument("-s", "--speed", type=float, default=1.0)
    p_speak.add_argument("-r", "--reference-audio")
    p_speak.add_argument("--prompt-text")
    p_speak.add_argument("--exaggeration", type=float, default=0.0)
    p_speak.add_argument("--variant", default="standard")
    p_speak.add_argument("--device", default="auto")

    # 5. PRELOAD COMMAND
    p_preload = subparsers.add_parser("preload", help="Preload model weights and assets locally")
    p_preload.add_argument("-e", "--engine", choices=["all", "kokoro", "chatterbox", "fish-speech"], default="all", help="Engine models to preload (default: all)")
    p_preload.add_argument("--variant", default="standard", help="Variant for Chatterbox (standard, turbo, multilingual)")
    p_preload.add_argument("--checkpoint", help="Specific checkpoint for Fish Speech")

    # 6. STATUS COMMAND
    subparsers.add_parser("status", help="Show local cache and model status")

    # 7. LIST-VOICES COMMAND
    p_voices = subparsers.add_parser("list-voices", help="List available preset voices")
    p_voices.add_argument("-e", "--engine", choices=["all"] + list_supported_engines(), default="all", help="Engine to query voices for (default: all)")

    return parser


def main() -> None:
    parser = build_parser()

    # If no arguments provided, print help
    if len(sys.argv) == 1:
        parser.print_help()
        sys.exit(0)

    # Shorthand support: if first arg is not a recognized subcommand and not a flag, default to 'kokoro'
    first_arg = sys.argv[1]
    recognized_cmds = {"kokoro", "chatterbox", "fish-speech", "fish", "speak", "preload", "status", "list-voices", "-h", "--help"}
    if first_arg not in recognized_cmds and not first_arg.startswith("-"):
        sys.argv.insert(1, "kokoro")

    args = parser.parse_args()

    if args.subcommand == "preload":
        handle_preload(args)
        return

    if args.subcommand == "status":
        handle_status(args)
        return

    if args.subcommand == "list-voices":
        handle_list_voices(args)
        return

    # Synthesis commands: kokoro, chatterbox, fish-speech, speak
    raw_text = getattr(args, "text", "")
    text_flag = getattr(args, "text_flag", None)
    file_arg = getattr(args, "file", None)
    text = read_input_text(text_flag or raw_text, file_arg)

    if not text:
        print("[Error] No input text provided. Provide text as an argument, via --file, or pipe via stdin.", file=sys.stderr)
        sys.exit(1)

    engine_name = getattr(args, "engine", args.subcommand)
    if engine_name in ("fish", "fish-speech"):
        engine_name = "fish-speech"

    output_path = getattr(args, "output", "output.wav")
    play = getattr(args, "play", False)

    # Extract all relevant engine kwargs from args namespace
    kwargs = {k: v for k, v in vars(args).items() if k not in ("subcommand", "text", "text_flag", "file", "output", "play", "engine") and v is not None}

    try:
        run_synthesis(
            engine_name=engine_name,
            text=text,
            output_path=output_path,
            play=play,
            **kwargs,
        )
    except Exception as e:
        print(f"\n[Synthesis Error] {e}", file=sys.stderr)
        sys.exit(1)


if __name__ == "__main__":
    main()
