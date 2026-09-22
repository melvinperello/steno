# AI Agent Instructions (`AGENTS.md`)

This document serves as the global context and instruction manual for any AI agent working on the Steno CLI. Read this carefully before modifying the codebase.

## Project Philosophy
Steno is a single-purpose, rock-solid CLI workhorse for transcribing long-form audio (up to 9+ hours). It explicitly prioritizes **absolute reliability and strict memory constraints** over raw throughput or aesthetics.

## Strict Architectural Constraints
When writing or modifying code in this repository, you MUST adhere to the following rules:

1. **Zero SQLite / No Databases**: 
   - Never introduce SQLite or heavy database ORMs. State must be managed purely via flat JSONL files (`checkpoint.py`) so users can easily read, modify, or delete their state.
   - Always use `os.fsync()` when writing state to guarantee resumability even during hard power failures.

2. **Constant ~50MB RAM Constraint**: 
   - NEVER load an entire audio file into memory. (e.g., do not use `librosa.load` or `soundfile.read` on the whole file).
   - Audio must be converted to a local `.wav` cache using FFmpeg, and streamed strictly in chunks using `wave.setpos()` in python.

3. **Transcription & Hallucination Prevention**:
   - Do NOT feed raw audio straight to Whisper.
   - You must strictly use the Silero VAD to detect contiguous speech blocks (merged up to 15s).
   - Whisper must only process isolated VAD segments. 
   - `condition_on_previous_text=False` MUST be set on the Whisper decoder to prevent infinite hallucination loops across silent boundaries.

4. **UI & Output**:
   - The CLI is powered exclusively by `click` and `rich`.
   - The UI must never block the main transcription loop. Any UI updating must be handled non-destructively (e.g. via background ticker threads) to accommodate blocking C++ inference engines.
   - Always support the `--json-stream` flag to bypass `rich` output entirely and emit NDJSON for external UI wrappers.

## Core Stack
- **Dependencies**: Managed strictly via `uv` in `pyproject.toml`.
- **Inference**: `faster-whisper` (CTranslate2) and `onnxruntime` (Silero VAD).
- **Hardware**: We detect NVIDIA GPUs strictly via `nvidia-smi` subprocess polling to avoid loading massive CUDA contexts on CPU-only machines.

## Development & Iteration Methodology
When collaborating on this project:
1. **Planning & Execution**: Implement features in accordance with the strict architectural constraints above.
2. **Guidelines & Pitfalls**: Refer to `docs/guidelines.md` for architectural standards and `docs/pitfalls.md` for recorded edge cases and lessons learned.
3. **Documentation & Ledger**: Keep `README.md` updated as the single source of truth for user-facing features, options, and architecture.
