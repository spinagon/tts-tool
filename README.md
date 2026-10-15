# Unified Text-to-Speech CLI (`tts`)

A fast, modular, offline-first command-line tool for Text-to-Speech (TTS), supporting:
1. **Kokoro-82M**: The ultra-lightweight speed king (~82M parameters, near-instant CPU & GPU inference).
2. **Chatterbox** (Resemble AI): Expressive zero-shot voice cloning, emotional exaggeration, and paralinguistic audio tags (`[laugh]`, `[sigh]`, `[cough]`).
3. **Fish Speech** (Fish Audio): High-fidelity multilingual zero-shot voice cloning and long-form audiobook narration via local checkpoints or Docker/HTTP API server.

---

## Key Features

* **Auto-Download at First Use**: Missing models, weights, and voice packs are detected and downloaded automatically upon first synthesis.
* **Explicit Preload Command**: Pre-download all or specific model checkpoints beforehand using `tts preload`.
* **Deep Parameter Control**: Exposes full model-specific configuration (emotions, guidance, speed, pauses, sampling, temperature, voice cloning).
* **Flexible Input & Output**: Accepts inline text, text files (`-f story.txt`), or stdin pipes (`cat text.txt | tts ...`). Outputs `.wav`, `.mp3`, `.ogg`, or `.flac`.
* **Instant Playback**: Add `--play` / `-p` to hear generated audio immediately.

---

## Quick Start

The tool comes ready with a launcher script (`./tts`):

```bash
# Display help and available subcommands
./tts --help

# Check status of local model checkpoints and cache
./tts status

# List preset voices
./tts list-voices --engine kokoro
```

---

### Synchronized Subtitles & Video Graphics Timestamps

To automate video creation with synced captions, karaoke highlighting, or graphic overlays, pass `--timestamps`:

```bash
# Generate audio alongside standard SRT subtitles (sentence-level and word-level)
./tts kokoro "Welcome to automated video generation. Subtitles are now perfectly synced!" \
  --timestamps srt \
  -o speech.wav

# Outputs generated:
#   speech.wav        (Audio waveform)
#   speech.srt        (Sentence-level subtitle cues)
#   speech.words.srt  (Individual word-by-word timing cues for kinetic typography)

# Generate JSON for programmatic video pipelines (Remotion, MoviePy, FFmpeg, After Effects)
./tts kokoro "Creating short-form video clips." --timestamps json -o short.wav

# Export all subtitle & timing formats simultaneously (JSON, SRT, WebVTT, CSV)
./tts kokoro "Full timeline export." --timestamps all -o timeline.wav
```

#### Timestamps Options
* `--timestamps [json|srt|vtt|csv|all]`: Export format (default: `json` if flag passed without value).
* `--timestamps-level [sentence|word|both]`: Granularity level (default: `both`).
* `--timestamps-file <path>`: Custom base file path for timing files.

---

### 1. Kokoro TTS (Fast CPU & GPU)

Generates speech instantly with preset voices:

```bash
# Basic synthesis using default voice (af_heart)
./tts kokoro "Hello! This is Kokoro running offline on local hardware." -o hello.wav --play

# Using a British male voice at 1.15x speed
./tts kokoro "Pipelines and builds have completed successfully." --voice bm_george --speed 1.15 -o george.wav

# Multilingual generation (Japanese, Mandarin, Spanish, French, Italian, Hindi, Portuguese)
./tts kokoro "こんにちは、世界！" --voice jf_alpha -o japanese.wav
./tts kokoro "你好，这是开源语音合成。" --voice zf_xiaobei -o chinese.wav

# Advanced prosody: continuous overlapping windows and custom sentence pauses
./tts kokoro "First thought. Second sentence." --sentence-pause 0.4 --continuous -o smooth.wav
```

### 2. Chatterbox TTS (Voice Cloning & Emotional Control)

High-fidelity zero-shot voice cloning with emotional control:

```bash
# Zero-shot voice cloning with ~10s reference audio
./tts chatterbox "Welcome back to our broadcast." --reference-audio sample.wav -o cloned.wav

# Emotional Exaggeration Control (0.0 = neutral, 0.7 = dynamic, 1.2 = dramatic)
./tts chatterbox "I cannot believe we actually pulled this off!" \
  --reference-audio sample.wav \
  --exaggeration 0.8 \
  -o dramatic.wav

# Chatterbox Turbo with paralinguistic cues
./tts chatterbox "Well [chuckle], that was certainly unexpected! [sigh]" \
  --variant turbo \
  -o funny.wav

# Multilingual variant
./tts chatterbox "Bonjour, comment ça va aujourd'hui?" \
  --variant multilingual \
  --language-id fr \
  --reference-audio french_sample.wav \
  -o french.wav
```

### 3. Fish Speech (Voice Cloning & Audiobooks)

State-of-the-art voice cloning with dual reference conditioning:

```bash
# Voice cloning with reference audio and transcript
./tts fish-speech "This is Fish Speech reproducing tone and cadence." \
  --reference-audio sample.wav \
  --prompt-text "Transcript of what is spoken inside sample.wav" \
  -o fish_cloned.wav

# Tuning generation parameters
./tts fish-speech -f chapter1.txt \
  --temperature 0.65 \
  --top-p 0.75 \
  --chunk-length 250 \
  -o chapter1.wav

# Connecting to a local Fish Speech Docker/Server instance
./tts fish-speech "Streaming via local server" --server-url http://localhost:8080 -o server_out.wav
```

---

## Preloading Models

To download models ahead of time (e.g. for air-gapped environments or pre-deployment):

```bash
# Preload all models
./tts preload --engine all

# Preload Kokoro assets only (~330 MB)
./tts preload --engine kokoro

# Preload Chatterbox weights
./tts preload --engine chatterbox --variant standard

# Preload Fish Speech checkpoints
./tts preload --engine fish-speech
```

---

## Parameter Reference

### Common Options (All Commands)
| Option | Description | Default |
| :--- | :--- | :--- |
| `text` | Positional text argument | `""` |
| `-t, --text-arg` | Text string argument flag | `None` |
| `-f, --file` | Path to text file to synthesize | `None` |
| `-o, --output` | Output audio path (`.wav`, `.mp3`, `.ogg`, `.flac`) | `output.wav` |
| `-p, --play` | Immediately play audio using system player (`ffplay`, `mpv`, `aplay`) | `False` |

### Kokoro Options (`tts kokoro`)
| Option | Description | Default |
| :--- | :--- | :--- |
| `-v, --voice` | Voice ID (e.g., `af_heart`, `am_adam`, `bf_emma`, `bm_george`, `zf_xiaobei`) | `af_heart` |
| `-s, --speed` | Speech pace multiplier (0.5 to 2.0) | `1.0` |
| `-l, --lang` | Language code (`auto`, `en-us`, `en-gb`, `ja`, `cmn`, `es`, `fr-fr`, `hi`, `it`, `pt-br`) | `auto` |
| `--backend` | Runtime engine (`onnx` or `pytorch`) | `onnx` |
| `--sentence-pause`| Pause length between sentences in seconds | `0.25` |
| `--clause-pause` | Pause length between clauses in seconds | `0.10` |
| `--continuous` | Overlapping window synthesis for smoother cross-sentence prosody | `False` |
| `--no-trim` | Preserve starting and ending audio padding | `False` |

### Chatterbox Options (`tts chatterbox`)
| Option | Description | Default |
| :--- | :--- | :--- |
| `--variant` | Model variant: `standard` (0.5B), `turbo` (350M), `multilingual` | `standard` |
| `-r, --reference-audio` | Audio file for zero-shot voice cloning (~5-15s WAV/MP3) | `None` |
| `-e, --exaggeration` | Emotional expressiveness slider (0.0 to 1.5+) | `0.0` |
| `--cfg-weight` | Classifier-Free Guidance weight for prompt adherence | `0.5` |
| `--temperature` | Sampling temperature | `0.8` |
| `--top-p` | Nucleus sampling probability threshold | `0.95` |
| `--language-id` | Multilingual code (`en`, `es`, `fr`, `de`, `it`, `zh`, `ja`, etc.) | `en` |
| `--device` | Compute device (`auto`, `cuda`, `cpu`, `mps`) | `auto` |

### Fish Speech Options (`tts fish-speech`)
| Option | Description | Default |
| :--- | :--- | :--- |
| `-r, --reference-audio` | Reference audio clip for zero-shot cloning | `None` |
| `--prompt-text` | Transcript of reference audio (vital for conditioning) | `None` |
| `--temperature` | Sampling temperature | `0.7` |
| `--top-p` | Nucleus sampling threshold | `0.7` |
| `--repetition-penalty` | Repetition penalty | `1.2` |
| `--max-new-tokens` | Maximum generated tokens | `1024` |
| `--chunk-length` | Chunk size for long text chunking | `200` |
| `-s, --speed` | Speed multiplier | `1.0` |
| `--server-url` | Server endpoint | `http://localhost:8080` |
| `--mode` | `auto`, `server`, `sdk`, or `local` | `auto` |

---

## Directory Structure & Architecture

```
tts_tool/
├── tts                     # Executable runner script
├── pyproject.toml          # Package configuration & dependencies
├── README.md               # Documentation
└── tts_cli/
    ├── __init__.py
    ├── cli.py              # CLI parser, subcommands, and dispatch
    ├── downloader.py       # Model cache & Hugging Face download manager
    ├── audio_utils.py      # Audio saving, encoding & playback utilities
    └── models/
        ├── __init__.py     # Engine registry
        ├── base.py         # BaseTTSEngine abstraction
        ├── kokoro_engine.py
        ├── chatterbox_engine.py
        └── fish_speech_engine.py
```
