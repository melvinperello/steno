import subprocess
import onnxruntime as ort

def has_nvidia_gpu() -> bool:
    """Check if an NVIDIA GPU is available and usable."""
    try:
        # The most reliable way to check for an NVIDIA GPU on Windows without 
        # initializing massive CUDA contexts is to just ask nvidia-smi.
        subprocess.check_output(['nvidia-smi'], stderr=subprocess.STDOUT)
        return True
    except (FileNotFoundError, subprocess.CalledProcessError):
        return False

def get_compute_config() -> tuple[str, str]:
    """
    Returns the optimal (device, compute_type) for the current hardware.
    As per Steno's memory-safe constraint, we default to int8 compute.
    """
    if has_nvidia_gpu():
        return ("cuda", "int8_float16") # fallback to int8_float16 for faster-whisper on some GPUs, or int8
    # If no GPU, fallback to CPU int8
    return ("cpu", "int8")
