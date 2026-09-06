import numpy as np
from faster_whisper import WhisperModel
from .hardware import get_compute_config
from .preprocess import read_wave_chunk

class Transcriber:
    """
    Core faster-whisper inference wrapper.
    Configures INT8 compute and transcribes audio chunks.
    """
    def __init__(self, model_size: str = "small"):
        device, compute_type = get_compute_config()
        # Initialize the model on the optimal hardware constraint
        self.model = WhisperModel(model_size, device=device, compute_type=compute_type)

    def transcribe_segment(self, wav_path: str, start_s: float, end_s: float, sample_rate: int = 16000) -> list[dict]:
        """
        Transcribes a specific audio segment using memory-safe numpy chunking.
        """
        start_sample = int(start_s * sample_rate)
        chunk_samples = int((end_s - start_s) * sample_rate)
        
        # Load only the required segment into RAM (keeps usage minimal)
        audio_array = read_wave_chunk(wav_path, start_sample, chunk_samples)
        
        if len(audio_array) == 0:
            return []

        # We disable condition_on_previous_text to prevent hallucinations across boundaries,
        # especially since we are feeding isolated VAD segments.
        segments, info = self.model.transcribe(
            audio_array, 
            beam_size=5, 
            vad_filter=False, # We already ran our own VAD
            condition_on_previous_text=False
        )
        
        for s in segments:
            # avg_logprob is typically negative. Convert to an approximate [0, 1] confidence.
            confidence = np.exp(s.avg_logprob)
            yield {
                "start": start_s + s.start,
                "end": start_s + s.end,
                "text": s.text.strip(),
                "confidence": float(confidence)
            }
