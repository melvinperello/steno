# Steno Architectural & Development Guidelines

This document outlines the strict engineering rules, architectural constraints, and standards for contributing to Steno. These principles guarantee that Steno remains a rock-solid, single-purpose CLI workhorse capable of processing long-form audio (up to 9+ hours) without memory leaks, deadlocks, or crashes.

---

## 1. Zero SQLite / No Databases

- **Flat JSONL Files Only**: Never introduce SQLite, DuckDB, or database ORMs. State must be maintained strictly in flat `.stcp.jsonl` files (`src/steno/checkpoint.py`).
- **User-Inspectable & Portable**: Checkpoint files must remain plain text so users can easily view, debug, modify, or delete them.
- **Mandatory `os.fsync()`**: An explicit `os.fsync(file_descriptor)` must be invoked after appending each segment. This guarantees zero state loss if power is cut or the system crashes.

---

## 2. Strict Constant ~50MB RAM Constraint

- **No Full-File Audio Loading**: NEVER load complete audio files or waveform arrays into memory (avoid `librosa.load()`, `soundfile.read()`, or loading entire PCM files).
- **Chunked Streaming via `wave.setpos()`**: Audio must be normalized to a local 16kHz mono WAV cache (`.st.wav`) using FFmpeg.
- **Sliding Window Reads**: Audio samples must be read strictly in 10-second windows using `wave.setpos(sample_offset)` and `wave.readframes(chunk_samples)`.
- **Memory Footprint**: RAM usage must stay flat around ~50MB regardless of whether the media file is 5 minutes or 9+ hours long.

---

## 3. Transcription & Hallucination Prevention

- **Never Feed Raw Audio Directly to Whisper**: Whisper decoders hallucinate when exposed to non-speech silence, ambient noise, or static over long periods.
- **Silero VAD Speech Gating**: Audio must first pass through Silero VAD (ONNX Runtime) to detect contiguous speech segments.
- **Segment Merging**: Individual speech frames must be merged into contiguous blocks up to 15 seconds long with a 0.2-second padding window.
- **Isolate Inference Context**: The Whisper decoding option `condition_on_previous_text=False` MUST be set. This isolates inferences to the current speech segment and prevents infinite hallucination loops across silent boundaries.

---

## 4. Sequential VRAM Lifecycle

- **Never Co-Allocate Models**: Silero ONNX and Faster-Whisper must never run concurrently on the GPU.
- **Lifecycle Sequence**:
  1. Initialize Silero VAD ONNX session (using `CUDAExecutionProvider` if an NVIDIA GPU is present).
  2. Stream and process the full WAV cache, outputting `.stvad.json`.
  3. Explicitly destroy the Silero session and invoke garbage collection to flush the CUDA context from GPU memory.
  4. Instantiate the Faster-Whisper model into clean VRAM.
- **Low-VRAM Protection**: This sequence physically shields entry-level consumer GPUs (4GB / 6GB VRAM) from out-of-memory exceptions during large batch jobs.

---

## 5. UI & CLI Output Guidelines

- **Asynchronous UI Ticker**: The CTranslate2 C++ engine locks the Python GIL during inference. Any UI updates (e.g. ETA, speed factor, timers) must run in a separate background daemon thread (`ui_ticker`) so the terminal never freezes.
- **Direct Rich Text Construction**: Do not use bracketed string markup for Rich progress bars. Construct `rich.text.Text` objects directly to prevent syntax collisions with progress counters (e.g. `[00:15 / 00:30]`).
- **Unicode Grapheme Hardening**: Always hard-truncate the live text variable (max 50 characters) at the Python string level before passing it to `rich.live.Live`. This prevents foreign scripts (Tamil, Arabic, Thai) from causing line-wrap cascades that break terminal cursor positioning.
- **Transient Batch Rendering**: Use `transient=True` on `rich.live.Live` panels when batching multiple files so completed files evaporate into clean 1-line log summaries.
- **Machine Integrations**: The `--json-stream` flag must suppress all terminal UI and emit pure NDJSON events to `stdout` for external integrations (Electron, Tauri, CLI wrappers).

---

## 6. Hardware Detection Standards

- **Subprocess Hardware Polling**: Do NOT rely on Python packages (such as `onnxruntime` or `torch.cuda.is_available()`) for hardware detection, as they frequently report false positives on Windows when CUDA libraries are installed without a physical GPU.
- **Direct `nvidia-smi` Check**: Check for an NVIDIA GPU strictly via `subprocess.check_output(['nvidia-smi'])`.
- **Quantization Fallbacks**: Default to `int8_float16` for CUDA devices and `int8` for CPU fallbacks.
