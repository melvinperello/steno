# Steno Pitfalls, Edge Cases & Lessons Learned

This document serves as an engineering ledger of critical pitfalls, edge cases, and hard-earned solutions discovered during the development and hardening of Steno.

---

## 1. C++ Engine UI Freeze (GIL Lock)

### The Pitfall
`faster-whisper` is powered by the `CTranslate2` C++ engine. When decoding an audio chunk, CTranslate2 holds the Python GIL, freezing the main Python thread—especially during CPU inference where decoding a segment can take several seconds. As a result, terminal progress bars, elapsed timers, and ETA indicators froze completely until inference finished.

### The Solution
We decoupled UI timing from the inference loop. A background daemon thread (`ui_ticker`) continuously calculates elapsed wall-clock time, rolling speed factors, and dynamic ETA predictions every 250ms, refreshing the `rich.live.Live` console while the main thread is locked in C++ computation.

---

## 2. False Positives in GPU Detection

### The Pitfall
On Windows machines with CUDA development packages or stale registry entries installed, libraries like `onnxruntime` or `torch` report `CUDAExecutionProvider` or CUDA capability as "available", even when running on an Intel/AMD CPU-only machine or when the physical GPU is disabled. Attempting to initialize CUDA in this state triggers cryptic DLL loading errors or crashes.

### The Solution
We bypass Python library assertions completely and poll the physical hardware via:
```python
subprocess.check_output(['nvidia-smi'], stderr=subprocess.STDOUT)
```
If `nvidia-smi` is not callable or fails, Steno immediately falls back to CPU execution without loading massive, unstable CUDA contexts.

---

## 3. Terminal Line-Wrap Cascade with Unicode Foreign Graphemes

### The Pitfall
When transcribing multilingual audio (such as Tamil, Arabic, Hindi, or Thai), complex multi-byte Unicode characters have ambiguous display widths in terminal emulators. Even with `rich` configured to `no_wrap=True` or `overflow="ellipsis"`, the physical terminal emulator wraps the line if it reaches the edge. When a line wraps, `rich.live.Live`'s ANSI cursor repositioning fails, resulting in stacked, glitching duplicate progress panels that permanently break the terminal screen.

### The Solution
We implemented a strict, defensive character limit at the raw Python string level:
```python
display_text = latest_text
if len(display_text) > 50:
    display_text = display_text[:47] + "..."
```
Because the string is truncated well before reaching the terminal margin, line-wrap is physically impossible regardless of font or terminal emulator implementation.

---

## 4. Concurrent VRAM Starvation on Consumer GPUs

### The Pitfall
Running Silero VAD (ONNX Runtime with CUDA) and Faster-Whisper simultaneously leads to memory contention. On 4GB or 6GB VRAM GPUs (common on laptop and budget developer setups), having both models resident in VRAM causes out-of-memory crashes on long audio files.

### The Solution: Sequential VRAM Allocation
Steno splits execution into strict, isolated phases:
1. Initialize Silero VAD session, stream the audio in 10s chunks, and export `.stvad.json`.
2. Terminate the Silero ONNX session and force garbage collection, returning 100MB+ of CUDA context to the operating system.
3. Only then instantiate Faster-Whisper in clean VRAM.
This dynamic lifecycle physically guarantees that batch runs across dozens of files never suffer from memory bloat.

---

## 5. Rich Markup Clashing with Literal Bracket Characters

### The Pitfall
Rich parses strings for style tags such as `[bold yellow]`. However, CLI progress bars commonly display literal brackets, such as `[00:15 / 00:30]`. Rich's string parser attempted to interpret `[00:15 / 00:30]` as a style tag, corrupting colors or stripping characters from the output.

### The Solution
We construct all UI lines explicitly using Python `Text` objects:
```python
prog_line = Text(f"  Progress : [{t_current} / {t_total}] [")
prog_line.append("█" * filled, style="bold yellow")
prog_line.append("░" * (bar_len - filled))
prog_line.append(f"] {percent:.1f}%")
```
This bypasses Rich's string markup parser entirely, guaranteeing deterministic color and formatting rendering.

---

## 6. The Whisper "Streaming" Illusion

### The Pitfall
It is tempting to assume Faster-Whisper can stream words token-by-token in real time. In practice, feeding a 15-second speech segment to Faster-Whisper forces the decoder to evaluate the entire segment atomically.

### The Solution
We structured `transcribe_segment()` as a Python `yield` generator. Segments are yielded and piped to disk checkpointing and live UI display the exact millisecond the C++ engine finishes each sentence, providing immediate feedback without artificial batch delays.

---

## 7. ONNX Runtime Provider Warning Clutter

### The Pitfall
When initializing `onnxruntime` with providers `["CUDAExecutionProvider", "CPUExecutionProvider"]`, the library emits noisy native Python `UserWarning` messages to `stderr` whenever evaluating hardware fallback paths. In a live terminal UI, these warnings disrupt ANSI cursor tracking and produce visual artifacts.

### The Solution
We wrapped the session instantiation in a warning filter:
```python
import warnings
with warnings.catch_warnings():
    warnings.simplefilter("ignore")
    session = ort.InferenceSession(str(model_path), sess_options=opts, providers=providers)
```
This keeps stdout and stderr completely clean for `rich.live.Live` and `--json-stream`.
