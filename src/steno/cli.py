import os
# Suppress HuggingFace symlink warnings on Windows
os.environ["HF_HUB_DISABLE_SYMLINKS_WARNING"] = "1"

import click
from pathlib import Path
from .pipeline import run_pipeline
from .checkpoint import CheckpointManager

@click.command()
@click.argument('path', type=click.Path(exists=True))
@click.option('--model', default='small', help='Whisper model size.')
@click.option('--min-confidence', default=0.6, help='Confidence threshold for filtering [UNCLEAR_AUDIO].')
@click.option('--clean', is_flag=True, help='Clean intermediate files and exit.')
@click.option('--retranscribe', is_flag=True, help='Force re-transcription from scratch (clears checkpoint).')
@click.option('--json-stream', is_flag=True, help='Suppress UI and emit NDJSON events for external integrations.')
def cli(path, model, min_confidence, clean, retranscribe, json_stream):
    """
    steno - Resumable, zero-bloat long-form audio transcription.
    
    PATH can be a single audio/video file or a directory of files.
    """
    target_path = Path(path)
    
    if target_path.is_file():
        files = [target_path]
    elif target_path.is_dir():
        exts = {'.mp3', '.m4a', '.wav', '.aac', '.flac', '.ogg', '.mp4', '.mkv', '.mov', '.webm'}
        files = [p for p in target_path.iterdir() if p.is_file() and p.suffix.lower() in exts]
        files = [p for p in files if not p.name.endswith('.st.wav')]
    else:
        click.echo("Invalid path.")
        return

    total_files = len(files)

    for idx, f in enumerate(files, 1):
        if clean:
            wav = f.with_suffix('.st.wav')
            vad = f.with_suffix('.stvad.json')
            cp = f.with_suffix('.stcp.jsonl')
            for cache in (wav, vad, cp):
                if cache.exists():
                    cache.unlink()
            if not json_stream: click.echo(f"Cleaned intermediate caches for {f.name}")
            continue
            
        if retranscribe:
            CheckpointManager(f).clear()
            
        # Skip already completed files if we aren't re-transcribing
        srt_path = f.with_suffix('.srt')
        if srt_path.exists() and not retranscribe:
            if not json_stream: click.echo(f"Skipping {f.name}, output SRT already exists.")
            continue
            
        try:
            batch_info = (idx, total_files) if total_files > 1 else None
            run_pipeline(f, model_size=model, min_confidence=min_confidence, batch_info=batch_info, json_stream=json_stream)
        except KeyboardInterrupt:
            if not json_stream: click.echo(f"\n[Steno] Interrupted by user. Progress for {f.name} was saved to its checkpoint.")
            break
        except Exception as e:
            if not json_stream: click.echo(f"Error processing {f.name}: {e}")

if __name__ == "__main__":
    cli()
