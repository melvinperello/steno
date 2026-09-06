import subprocess
import wave
from pathlib import Path
import numpy as np

def get_ffmpeg_path() -> str:
    """Check for local ffmpeg.exe in the repo before falling back to system PATH."""
    repo_root = Path(__file__).parent.parent.parent
    for local_path in [repo_root / "ffmpeg.exe", repo_root / "bin" / "ffmpeg.exe"]:
        if local_path.exists():
            return str(local_path)
    return "ffmpeg"

def prep_audio(input_path: str | Path) -> Path:
    """
    Converts audio to 16kHz mono WAV (Tier 0 Cache) if it doesn't already exist.
    Uses FFmpeg via subprocess.
    """
    input_path = Path(input_path)
    wav_path = input_path.with_suffix('.st.wav')
    
    if wav_path.exists():
        return wav_path

    cmd = [
        get_ffmpeg_path(),
        "-i", str(input_path),
        "-ar", "16000",
        "-ac", "1",
        "-c:a", "pcm_s16le",
        "-y", 
        str(wav_path)
    ]
    
    try:
        subprocess.run(cmd, check=True, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
    except FileNotFoundError:
        raise RuntimeError("ffmpeg not found. Please ensure ffmpeg is installed and in your PATH.")
    except subprocess.CalledProcessError as e:
        if wav_path.exists():
            wav_path.unlink()
        raise RuntimeError(f"FFmpeg failed to process audio: {e}")
        
    return wav_path

def get_audio_duration(wav_path: str | Path) -> float:
    """Returns duration in seconds of the cached wav."""
    with wave.open(str(wav_path), "rb") as wf:
        return wf.getnframes() / wf.getframerate()

def read_wave_chunk(wav_path: str | Path, start_sample: int, chunk_samples: int) -> np.ndarray:
    """
    Memory-safe reading of a specific chunk of the WAV file.
    Maintains ~50MB RAM by only loading requested samples.
    """
    with wave.open(str(wav_path), "rb") as wf:
        nframes = wf.getnframes()
        if start_sample >= nframes:
            return np.array([], dtype=np.float32)
            
        wf.setpos(start_sample)
        # Cap read to remaining frames
        frames_to_read = min(chunk_samples, nframes - start_sample)
        raw_bytes = wf.readframes(frames_to_read)
        
        # Convert 16-bit PCM bytes to float32 [-1.0, 1.0]
        samples = np.frombuffer(raw_bytes, dtype=np.int16).astype(np.float32) / 32768.0
        return samples
