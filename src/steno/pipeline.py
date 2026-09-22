import time
import json
from pathlib import Path
from rich.live import Live
from rich.text import Text
from rich.panel import Panel
from rich.layout import Layout
from rich.console import Group

from .preprocess import prep_audio, get_audio_duration
from .vad import generate_vad_cache
from .checkpoint import CheckpointManager
from .transcribe import Transcriber
from .hardware import get_compute_config, has_nvidia_gpu
from .output import generate_outputs

def format_time(seconds: float) -> str:
    h = int(seconds // 3600)
    m = int((seconds % 3600) // 60)
    s = int(seconds % 60)
    if h > 0:
        return f"{h:02d}:{m:02d}:{s:02d}"
    return f"{m:02d}:{s:02d}"

def generate_ui(file_name, duration, current, speed, elapsed, eta, latest_text, engine_info, status, batch_info=None, segment_info=None):
    percent = (current / duration * 100) if duration > 0 else 0
    bar_len = 24
    filled = int((percent / 100) * bar_len)
    
    t_current = format_time(current)
    t_total = format_time(duration)
    
    lines = []
    if batch_info:
        b_idx, b_tot = batch_info
        b_pct = (b_idx / b_tot) * 100
        b_filled = int((b_pct / 100) * bar_len)
        
        batch_line = Text(f"  Files    : [{b_idx}/{b_tot}] [")
        batch_line.append("█" * b_filled, style="bold yellow")
        batch_line.append("░" * (bar_len - b_filled))
        batch_line.append(f"] {int(b_pct)}% • {file_name}")
        lines.append(batch_line)
        lines.append(Text(""))
        
    lines.extend([
        Text(f"  File     : {file_name} ({t_total})"),
        Text(f"  Engine   : {engine_info}"),
        Text(f"  Status   : {status}"),
        Text("")
    ])
    
    seg_str = f" • Seg {segment_info[0]}/{segment_info[1]}" if segment_info else ""
    
    prog_line = Text(f"  Progress : [{t_current} / {t_total}] [")
    prog_line.append("█" * filled, style="bold yellow")
    prog_line.append("░" * (bar_len - filled))
    prog_line.append(f"] {percent:.1f}%{seg_str}")
    lines.append(prog_line)
    
    # Hard-truncate the live text to guarantee it never touches the right edge of the terminal,
    # as complex Unicode graphemes completely break terminal width calculations.
    display_text = latest_text
    if len(display_text) > 50:
        display_text = display_text[:47] + "..."
        
    lines.extend([
        Text(f"  Speed    : {speed:.1f}x real-time • Elapsed: {format_time(elapsed)} • ETA: {format_time(eta)}"),
        Text(f"  Live     : 💬 {display_text}", no_wrap=True, overflow="ellipsis")
    ])
    
    return Panel(Group(*lines), title="[bold yellow]steno[/bold yellow]", title_align="left", border_style="bold yellow")

import contextlib

@contextlib.contextmanager
def manage_ui(json_stream, initial_ui):
    if json_stream:
        yield None
    else:
        with Live(initial_ui, refresh_per_second=4, transient=True) as live:
            yield live

def emit_json(file_name, duration, current, speed, elapsed, eta, latest_text, status):
    percent = (current / duration * 100) if duration > 0 else 0
    print(json.dumps({
        "event": "progress",
        "file": file_name,
        "audio_duration_s": duration,
        "current_audio_s": current,
        "percent": percent,
        "speed_factor": speed,
        "elapsed_s": elapsed,
        "eta_s": eta,
        "latest_text": latest_text,
        "status": status
    }), flush=True)

def run_pipeline(audio_file: Path, model_size: str = "small", min_confidence: float = 0.6, batch_info=None, json_stream=False):
    start_wallclock = time.time()
    
    device, compute_type = get_compute_config()
    engine_info = f"Whisper {model_size} ({compute_type}) on {device.upper()}"
    status = "Initializing..."
    latest_text = "\"...\""
    duration = 0.0
    completed_s = 0.0
    
    def update_state(current, speed=0.0, elapsed=0.0, eta=0.0, l_text=None, stat=None, live_ctx=None, seg_info=None):
        nonlocal latest_text, status
        if l_text: latest_text = l_text
        if stat: status = stat
        
        if json_stream:
            emit_json(audio_file.name, duration, current, speed, elapsed, eta, latest_text, status)
        elif live_ctx:
            ui = generate_ui(audio_file.name, duration, current, speed, elapsed, eta, latest_text, engine_info, status, batch_info, seg_info)
            live_ctx.update(ui)
            
    initial_ui = generate_ui(audio_file.name, duration, completed_s, 0.0, 0.0, 0.0, latest_text, engine_info, status, batch_info)
    
    with manage_ui(json_stream, initial_ui) as live:
        # 1. Preprocess
        update_state(completed_s, stat="Converting Audio (FFmpeg)...", live_ctx=live)
        wav_path = prep_audio(audio_file)
        duration = get_audio_duration(wav_path)
        
        # 2. VAD
        def vad_progress(p: float):
            hw_str = "CUDA" if has_nvidia_gpu() else "CPU"
            update_state(completed_s, stat=f"Running Silero VAD ({hw_str})... {p*100:.1f}%", live_ctx=live)
            
        vad_cache_path = generate_vad_cache(wav_path, progress_callback=vad_progress)
        with open(vad_cache_path, 'r') as f:
            vad_segments = json.load(f)
            
        total_segs = len(vad_segments)
            
        # 3. Checkpoint
        checkpoint = CheckpointManager(audio_file)
        completed = checkpoint.load_completed_segments()
        completed_segs_count = 0
        if completed:
            completed_s = completed[-1]["end"]
            completed_segs_count = len([s for s in vad_segments if s["end"] <= completed_s])
            update_state(completed_s, stat=f"Resumed from checkpoint ({format_time(completed_s)} already done)", live_ctx=live, seg_info=(completed_segs_count, total_segs))
        else:
            update_state(completed_s, stat="Started new transcription", live_ctx=live, seg_info=(0, total_segs))
            
        # 4. Load Whisper
        status_text = "Downloading/Loading Whisper Model..."
        update_state(completed_s, stat=status_text, live_ctx=live, seg_info=(completed_segs_count, total_segs))
        
        import threading
        transcriber_result = []
        def load_model():
            transcriber_result.append(Transcriber(model_size=model_size))
            
        t = threading.Thread(target=load_model)
        t.start()
        
        load_start = time.time()
        while t.is_alive():
            elapsed_load = int(time.time() - load_start)
            update_state(completed_s, stat=f"{status_text} ({elapsed_load}s)", live_ctx=live, seg_info=(completed_segs_count, total_segs))
            time.sleep(0.2)
            
        transcriber = transcriber_result[0]
        update_state(completed_s, stat="Transcribing...", live_ctx=live, seg_info=(completed_segs_count, total_segs))
        
        # 5. Transcribe
        remaining_vad = [s for s in vad_segments if s["end"] > completed_s]
        
        shared_state = {
            "current_s": completed_s,
            "transcribed_audio_s": 0.0,
            "running": True,
            "seg_idx": completed_segs_count
        }
        
        def ui_ticker():
            while shared_state["running"]:
                elapsed = time.time() - start_wallclock
                cur = shared_state["current_s"]
                t_audio = shared_state["transcribed_audio_s"]
                s_idx = shared_state["seg_idx"]
                
                speed = t_audio / elapsed if elapsed > 0 else 0
                eta = (duration - cur) / speed if speed > 0 else 0
                
                update_state(cur, speed, elapsed, eta, stat="Transcribing...", live_ctx=live, seg_info=(s_idx, total_segs))
                time.sleep(0.25)
                
        ticker = threading.Thread(target=ui_ticker)
        if not json_stream:
            ticker.start()
        
        try:
            for idx, v_seg in enumerate(remaining_vad):
                current_seg_idx = completed_segs_count + idx + 1
                shared_state["seg_idx"] = current_seg_idx
                
                seg_start = v_seg["start"]
                seg_end = v_seg["end"]
                
                results = transcriber.transcribe_segment(wav_path, seg_start, seg_end)
                
                last_res_end = seg_start
                for res in results:
                    checkpoint.append_segment(res)
                    
                    # Update shared state for the ticker
                    chunk_progress = res['end'] - last_res_end
                    shared_state["transcribed_audio_s"] += chunk_progress
                    shared_state["current_s"] = res['end']
                    last_res_end = res['end']

                    elapsed = time.time() - start_wallclock
                    speed = shared_state["transcribed_audio_s"] / elapsed if elapsed > 0 else 0.0
                    eta = (duration - shared_state["current_s"]) / speed if speed > 0 else 0.0

                    l_text = f"[{format_time(res['start'])}] \"{res['text']}\""
                    update_state(res['end'], speed=speed, elapsed=elapsed, eta=eta, stat="Transcribing...", l_text=l_text, live_ctx=None, seg_info=(current_seg_idx, total_segs))
                
                # Catch up any remaining silence at the end of the VAD block
                remaining = seg_end - last_res_end
                if remaining > 0:
                    shared_state["transcribed_audio_s"] += remaining
                
                shared_state["current_s"] = seg_end
        finally:
            shared_state["running"] = False
            if not json_stream:
                ticker.join()
            
            # Final UI update
            elapsed = time.time() - start_wallclock
            speed = shared_state["transcribed_audio_s"] / elapsed if elapsed > 0 else 0
            eta = (duration - shared_state["current_s"]) / speed if speed > 0 else 0
            update_state(shared_state["current_s"], speed, elapsed, eta, stat="Transcribing...", live_ctx=live, seg_info=(shared_state["seg_idx"], total_segs))
            
    checkpoint.close()
    
    srt, csv_file = generate_outputs(checkpoint.checkpoint_path, min_confidence)
    if json_stream:
        print(json.dumps({"event": "complete", "file": audio_file.name, "output_file": str(srt)}), flush=True)
    else:
        print(f"Finished! Outputs saved to {srt} and {csv_file}")
