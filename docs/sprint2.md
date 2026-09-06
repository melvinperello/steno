# Sprint 2: UI Hardening, VAD Acceleration & Documentation

**Goal**: Polish the core engine by addressing edge-case terminal rendering bugs, accelerating the VAD phase with CUDA, and overhauling the repository documentation to serve as a permanent standard for future agents.

## Tasks Executed

1. **Terminal Cascade Bug Fix**: 
   - Addressed a severe terminal UI glitch where complex Unicode graphemes (e.g., Tamil) caused incorrect width calculations in `rich`, resulting in a cascading UI rendering bug.
   - *Fix*: Implemented a strict Python-level hard-truncate (max 50 characters) on the `latest_text` variable before it reaches `rich`, guaranteeing it never wraps the terminal buffer.

2. **Silero VAD Hardware Acceleration**: 
   - Refactored `vad.py` to dynamically load `CUDAExecutionProvider` if an NVIDIA GPU is detected via `nvidia-smi`.
   - Updated the `pipeline.py` UI string to explicitly report `Running Silero VAD (CUDA)...` instead of a generic GPU label.

3. **Documentation Overhaul**: 
   - Rewrote `AGENTS.md` from a temporary sprint tracker into a permanent, timeless System Prompt (defining strict memory and architecture constraints).
   - Added an "Architecture & Philosophy" section to `README.md` explaining the Whisper/VAD synergy and the intentional Sequential VRAM Allocation design.
   - Added `LICENSE.md` (MIT License).

---

## Sprint 2 Retrospective

### 🏆 What Worked Beautifully (The Wins)
1. **Sequential VRAM Allocation Justification**: We confirmed that loading/unloading the Silero ONNX Session dynamically inside `generate_vad_cache()` is mathematically optimal. It guarantees that the 100MB+ CUDA context used by Silero is fully destroyed and garbage collected before the massive Whisper model loads, physically protecting 4GB/6GB GPUs from out-of-memory crashes.
2. **Architecture Documentation**: Putting the exact philosophy (Zero SQLite, wave.setpos() streaming, condition_on_previous_text=False) into `AGENTS.md` guarantees that no future AI will accidentally "optimize" our engine into a memory leak.

### ⚠️ Pitfalls & Hard Lessons (The Challenges)
1. **Unicode Terminal Widths**: We learned that you absolutely cannot trust a terminal emulator to correctly calculate the column width of complex foreign graphemes (like Tamil or Arabic). Even with `rich` configured to `no_wrap=True`, the physical terminal will hard-wrap ambiguous width characters, breaking `rich.live.Live`'s vertical cursor tracking. The only bulletproof solution is a strict `len(str)` chop at the raw Python string level.
