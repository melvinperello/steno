# steno

A single-purpose, rock-solid CLI workhorse for transcribing long-form audio (up to 9+ hours) with zero RAM bloat, crash-proof JSONL resumability, and dead-simple single-file or directory batching.

## Features

- **Zero SQLite**: Resumability powered by pure JSONL files.
- **Constant ~50MB RAM**: Memory-safe WAV streaming using `wave.setpos()`.
- **Live UI**: Rich-powered "Proof-of-Life" timeline, progress bars, and ETA.
- **Pure Output**: Generates `.srt` and `.csv` files natively.

## Architecture & Philosophy

Steno achieves high accuracy and infinite file-length reliability by strictly decoupling Voice Activity Detection (VAD) from Transcription:

1. **Memory Safety**: We never load full audio arrays into RAM. Audio is streamed in 10-second chunks using `wave.setpos()`.
2. **Context & Accuracy**: The Silero VAD model detects continuous speech and merges it into solid blocks up to 15 seconds long. This provides Whisper with perfect grammatical context for highly accurate inferences, avoiding the pitfalls of blind, word-by-word chunking.
3. **Zero Hallucinations**: Because we only feed Whisper verified speech boundaries, it is never exposed to dead air or static, effectively eliminating the hallucination loops that commonly plague raw Whisper wrappers on long audio files.
4. **Sequential VRAM Allocation**: Both the Silero VAD model and the Whisper transcription model are explicitly instantiated and destroyed *per-file*. This completely flushes the GPU's memory between phases and between files in a batch, guaranteeing maximum VRAM availability and completely eliminating the creeping memory leaks that cause C++ frameworks to crash on massive, multi-hour batch runs.

## Installation

Steno requires Python 3.10+ and [uv](https://github.com/astral-sh/uv) for dependency management.

You must also have **FFmpeg** installed on your system or placed in the project folder. On Windows, you can easily install it using winget:

```powershell
winget install gyan.ffmpeg -v 7.0.2
```
*(Note: Restart your terminal after installation so FFmpeg is added to your PATH, or place `ffmpeg.exe` directly in this repo directory).*

To install Steno's Python dependencies:
```bash
uv sync
```

### Notes on First Run & CPU Performance
- **Model Downloads**: On the very first run, Steno will download the Silero VAD and Whisper models to your `~/.cache` directory. You may see harmless `huggingface_hub` warnings during this time (e.g. about symlinks or HF_TOKENs)—these can be safely ignored and won't appear on subsequent runs.
- **CPU Initialization**: If you do not have an NVIDIA GPU, loading the transcription models directly into CPU RAM can take 10-30 seconds. The Live terminal UI will explicitly state "Downloading/Loading Whisper Model..." while this occurs.

## Usage

Steno can process a single file or batch process an entire directory of audio/video files.

```bash
# Transcribe a single file
uv run steno podcast.m4a

# Batch transcribe an entire directory
uv run steno ./audio_folder/
```

### Command Line Options

| Flag | Default | Description |
|---|---|---|
| `--model TEXT` | `small` | The faster-whisper model size to use (`tiny`, `base`, `small`, `medium`, `large-v3`). |
| `--min-confidence FLOAT` | `0.6` | Words below this threshold are automatically replaced with `[UNCLEAR_AUDIO]` in the final output. |
| `--retranscribe` | `False` | Forces Steno to ignore any existing checkpoints and re-transcribe the file from scratch. |
| `--clean` | `False` | Deletes all intermediate cache files (`.st.wav`, `.stvad.json`, `.stcp.jsonl`) for the given path and exits. |
| `--json-stream` | `False` | Completely suppresses the Rich terminal UI and emits raw NDJSON events to `stdout` for easy integration into external apps (Electron, Tauri, etc.). |
