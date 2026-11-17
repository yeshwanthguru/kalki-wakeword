"""Listen on the microphone and print when 'hey kalki' is heard.

    python live.py [--threshold 0.8] [--model models/kalki_int8.onnx]
"""
import argparse
import time

import sounddevice as sd

from detector import DEFAULT_MODEL, HOP, SR, WakeWordDetector


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default=str(DEFAULT_MODEL))
    ap.add_argument("--threshold", type=float, default=0.8)
    ap.add_argument("--hits", type=int, default=2, help="consecutive windows above threshold")
    a = ap.parse_args()
    det = WakeWordDetector(a.model, a.threshold, a.hits)

    def cb(indata, frames, t, status):
        for hit in det.push(indata[:, 0]):
            print(f"WAKE  (p={hit.prob:.2f})", flush=True)

    with sd.InputStream(samplerate=SR, channels=1, blocksize=HOP, dtype="float32", callback=cb):
        print("listening... say 'hey kalki' (Ctrl+C to quit)")
        while True:
            time.sleep(0.5)


if __name__ == "__main__":
    main()
