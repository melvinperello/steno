import csv
from pathlib import Path

def format_srt_time(seconds: float) -> str:
    """Formats seconds to SRT timestamp format HH:MM:SS,mmm"""
    hours = int(seconds // 3600)
    minutes = int((seconds % 3600) // 60)
    secs = int(seconds % 60)
    millis = int((seconds - int(seconds)) * 1000)
    return f"{hours:02d}:{minutes:02d}:{secs:02d},{millis:03d}"

def generate_outputs(checkpoint_path: Path, min_confidence: float = 0.6):
    """
    Reads the pure JSONL checkpoint and generates strict .srt and .csv outputs.
    Applies the confidence filter.
    """
    import json
    
    if not checkpoint_path.exists():
        return

    audio_path = checkpoint_path.with_suffix('')
    # Handle the .stcp removal safely
    base_name = str(audio_path).replace('.stcp', '')
    srt_path = Path(f"{base_name}.srt")
    csv_path = Path(f"{base_name}.csv")

    segments = []
    with open(checkpoint_path, 'r', encoding='utf-8') as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                segments.append(json.loads(line))
            except json.JSONDecodeError:
                continue

    # Write SRT
    with open(srt_path, 'w', encoding='utf-8') as f:
        for idx, seg in enumerate(segments, 1):
            text = seg['text']
            if seg.get('confidence', 1.0) < min_confidence:
                text = "[UNCLEAR_AUDIO]"
            
            start_str = format_srt_time(seg['start'])
            end_str = format_srt_time(seg['end'])
            
            f.write(f"{idx}\n")
            f.write(f"{start_str} --> {end_str}\n")
            f.write(f"{text}\n\n")

    # Write CSV
    with open(csv_path, 'w', encoding='utf-8', newline='') as f:
        writer = csv.writer(f)
        writer.writerow(['start', 'end', 'text', 'confidence'])
        for seg in segments:
            text = seg['text']
            if seg.get('confidence', 1.0) < min_confidence:
                text = "[UNCLEAR_AUDIO]"
            writer.writerow([seg['start'], seg['end'], text, seg.get('confidence', 1.0)])
            
    return srt_path, csv_path
