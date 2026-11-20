"""'hey kalki' wake word detector: a 36 KB ONNX model with a streaming detector, CPU only.

    from kalki_wakeword import WakeWordDetector
    det = WakeWordDetector(sensitivity="balanced")      # or "sensitive" / "strict", or threshold= and hits=
    for hit in det.push(samples):            # float32 mono 16 kHz in [-1, 1], any chunk size
        print(f"heard it at {hit.time:.2f}s (p={hit.prob:.2f})")

Check an install on a recording:  python -m kalki_wakeword clip.wav
"""
from .detector import DEFAULT_MODEL, HOP, INT4_MODEL, SENSITIVITY, SR, WIN, Detection, WakeWordDetector

__version__ = "0.2.0"
__all__ = ["WakeWordDetector", "Detection", "DEFAULT_MODEL", "INT4_MODEL", "SENSITIVITY", "SR", "WIN", "HOP", "__version__"]
