"""python -m kalki_wakeword clip.wav [--threshold 0.8] [--hits 2]: run the detector over a 16 kHz 16-bit wav file."""
import argparse
import sys
import wave

import numpy as np

from . import SR, WakeWordDetector, __version__


def read_wav(path):
    """16 kHz PCM16 wav (any number of channels) as mono float32 in [-1, 1]."""
    with wave.open(str(path), "rb") as w:
        if w.getsampwidth() != 2:
            raise SystemExit(f"{path}: need 16-bit PCM wav (got {8 * w.getsampwidth()}-bit)")
        if w.getframerate() != SR:
            raise SystemExit(f"{path}: need {SR} Hz audio (got {w.getframerate()} Hz); resample it first, e.g. with ffmpeg -ar {SR}")
        x = np.frombuffer(w.readframes(w.getnframes()), dtype="<i2").astype(np.float32) / 32768.0
        return x.reshape(-1, w.getnchannels()).mean(1)


def main(argv=None):
    ap = argparse.ArgumentParser(prog="python -m kalki_wakeword", description=__doc__)
    ap.add_argument("wav", help="16 kHz, 16-bit PCM wav file")
    ap.add_argument("--threshold", type=float, default=0.8)
    ap.add_argument("--hits", type=int, default=2, help="consecutive windows above the threshold needed to fire")
    ap.add_argument("--version", action="version", version=f"kalki-wakeword {__version__}")
    a = ap.parse_args(argv)
    x = read_wav(a.wav)
    det = WakeWordDetector(threshold=a.threshold, hits=a.hits)
    hits = det.push(x) + det.push(np.zeros(4000, np.float32))   # let the last window finish
    for h in hits:
        print(f"heard it at {h.time:.2f}s (p={h.prob:.2f})")
    print(f"{len(hits)} detection(s) in {len(x) / SR:.1f}s of audio")
    return 0 if hits else 1


if __name__ == "__main__":
    sys.exit(main())
