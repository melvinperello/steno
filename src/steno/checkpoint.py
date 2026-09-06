import json
import os
from pathlib import Path
from typing import Any, Dict, List

class CheckpointManager:
    """
    Pure JSONL file manager that appends segments with `os.fsync()`.
    Provides a zero-loss resumability guarantee without SQLite bloat.
    """
    def __init__(self, audio_path: str | Path):
        self.audio_path = Path(audio_path)
        # We save the checkpoint as {name}.stcp.jsonl alongside the media file
        self.checkpoint_path = self.audio_path.with_suffix('.stcp.jsonl')
        self._file_handle = None

    def load_completed_segments(self) -> List[Dict[str, Any]]:
        """Loads successfully transcribed segments, ignoring trailing corruption."""
        if not self.checkpoint_path.exists():
            return []
        
        completed = []
        with open(self.checkpoint_path, 'r', encoding='utf-8') as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                try:
                    completed.append(json.loads(line))
                except json.JSONDecodeError:
                    # Power loss might corrupt the last written line; ignore it
                    continue
        return completed

    def append_segment(self, segment_data: Dict[str, Any]):
        """Appends a segment and issues an immediate fsync."""
        if self._file_handle is None:
            self._file_handle = open(self.checkpoint_path, 'a', encoding='utf-8')
        
        line = json.dumps(segment_data) + '\n'
        self._file_handle.write(line)
        self._file_handle.flush()
        # Guarantee write to disk to prevent data loss on crash
        os.fsync(self._file_handle.fileno())

    def close(self):
        """Closes the underlying file handle."""
        if self._file_handle is not None:
            self._file_handle.close()
            self._file_handle = None

    def clear(self):
        """Removes the checkpoint file entirely (for re-transcription)."""
        self.close()
        if self.checkpoint_path.exists():
            try:
                self.checkpoint_path.unlink()
            except OSError:
                pass

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        self.close()
