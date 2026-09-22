import json
import urllib.request
from pathlib import Path
import numpy as np
import onnxruntime as ort
from .preprocess import read_wave_chunk, get_audio_duration

SILERO_ONNX_URL = "https://github.com/snakers4/silero-vad/raw/master/src/silero_vad/data/silero_vad.onnx"
SAMPLE_RATE    = 16000
WINDOW_SAMPLES = 512    # ~32ms per inference window — must be exactly 512 for 16kHz
CONTEXT_SIZE   = 64     # samples prepended to each window (required by model)

def get_silero_model_path() -> Path:
    cache_dir = Path.home() / ".cache" / "steno"
    cache_dir.mkdir(parents=True, exist_ok=True)
    model_path = cache_dir / "silero_vad.onnx"
    
    if not model_path.exists():
        urllib.request.urlretrieve(SILERO_ONNX_URL, model_path)
    return model_path

def merge_segments(segments: list[dict], total_duration: float, merge_gap=0.3, min_length=0.5, max_length=15.0, padding=0.2) -> list[dict]:
    """Merges contiguous speech frames into unified segments."""
    if not segments:
        return []

    merged = []
    cur_start = segments[0]["start"]
    cur_end   = segments[0]["end"]
    cur_probs = [segments[0]["speech_prob"]]

    for seg in segments[1:]:
        gap = seg["start"] - cur_end
        would_be_length = seg["end"] - cur_start

        if gap <= merge_gap and would_be_length <= max_length:
            cur_end = seg["end"]
            cur_probs.append(seg["speech_prob"])
        else:
            if cur_end - cur_start >= min_length:
                merged.append({
                    "start": max(0.0, cur_start - padding),
                    "end": min(total_duration, cur_end + padding)
                })
            cur_start = seg["start"]
            cur_end   = seg["end"]
            cur_probs = [seg["speech_prob"]]

    if cur_end - cur_start >= min_length:
        merged.append({
            "start": max(0.0, cur_start - padding),
            "end": min(total_duration, cur_end + padding)
        })
    return merged

def generate_vad_cache(wav_path: Path, progress_callback=None) -> Path:
    """
    Runs Silero VAD over the WAV file strictly with constant memory footprint.
    Reads 10-second chunks into memory, processes them, and clears them.
    Saves and returns the path to `{name}.stvad.json`.
    """
    vad_cache = Path(str(wav_path).replace('.st.wav', '.stvad.json'))
    if vad_cache.exists():
        return vad_cache

    model_path = get_silero_model_path()
    from .hardware import has_nvidia_gpu
    providers = ["CUDAExecutionProvider", "CPUExecutionProvider"] if has_nvidia_gpu() else ["CPUExecutionProvider"]
    
    opts = ort.SessionOptions()
    opts.log_severity_level = 3
    
    import warnings
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        session = ort.InferenceSession(str(model_path), sess_options=opts, providers=providers)
    
    _state = np.zeros((2, 1, 128), dtype=np.float32)
    _context = np.zeros((1, CONTEXT_SIZE), dtype=np.float32)
    
    duration = get_audio_duration(wav_path)
    total_samples = int(duration * SAMPLE_RATE)
    chunk_size = 10 * SAMPLE_RATE # 10 seconds at a time (160k samples)
    
    raw_segments = []
    threshold = 0.6
    
    # Process audio chunk by chunk (memory-safe streaming)
    for chunk_start in range(0, total_samples, chunk_size):
        audio_chunk = read_wave_chunk(wav_path, chunk_start, chunk_size)
        
        for i in range(0, len(audio_chunk) - WINDOW_SAMPLES + 1, WINDOW_SAMPLES):
            win = audio_chunk[i:i + WINDOW_SAMPLES].reshape(1, -1)
            x = np.concatenate([_context, win], axis=1)  # (1, 576)

            ort_inputs = {
                "input": x,
                "state": _state,
                "sr": np.array(SAMPLE_RATE, dtype=np.int64),
            }

            out, _state = session.run(None, ort_inputs)
            _context = x[:, -CONTEXT_SIZE:]
            prob = float(out.squeeze())
            
            absolute_sample = chunk_start + i
            if prob >= threshold:
                start_s = absolute_sample / SAMPLE_RATE
                end_s = (absolute_sample + WINDOW_SAMPLES) / SAMPLE_RATE
                raw_segments.append({"start": start_s, "end": end_s, "speech_prob": prob})
                
        if progress_callback:
            progress_callback(min(1.0, (chunk_start + chunk_size) / total_samples))

    merged = merge_segments(raw_segments, duration)
    with open(vad_cache, 'w', encoding='utf-8') as f:
        json.dump(merged, f)
    
    return vad_cache
