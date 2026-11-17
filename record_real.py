"""Record your own voice so the detector learns it (training only ever heard synthetic speech).

    python record_real.py positive -n 40         # say "hey kalki" after each beep prompt
    python record_real.py negative --minutes 5   # just talk, play TV, type, run the fan: normal life
Then retrain:  python train.py --steps 6000
Clips land in data/positive/pos_real_N.wav and data/negative/neg_real_N.wav (16 kHz mono).
"""
import argparse
import time
from pathlib import Path

import numpy as np
import sounddevice as sd
import soundfile as sf

SR = 16000
DATA = Path(__file__).parent / "data"


def rec(seconds):
    x = sd.rec(int(seconds * SR), samplerate=SR, channels=1, dtype="float32")
    sd.wait()
    return x[:, 0]


def trim(a, pad=int(0.15 * SR)):
    idx = np.where(np.abs(a) > 0.05 * max(np.abs(a).max(), 1e-6))[0]
    return a[max(idx[0] - pad, 0): idx[-1] + pad] if len(idx) else a


def next_index(folder, tag):
    used = [int(p.stem.rsplit("_", 1)[1]) for p in folder.glob(f"{tag}_real_*.wav")]
    return max(used, default=-1) + 1


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("kind", choices=["positive", "negative"])
    ap.add_argument("-n", type=int, default=40, help="positive: number of takes")
    ap.add_argument("--minutes", type=float, default=5, help="negative: total minutes")
    a = ap.parse_args()
    out = DATA / a.kind
    out.mkdir(parents=True, exist_ok=True)
    tag = "pos" if a.kind == "positive" else "neg"
    k = next_index(out, tag)

    if a.kind == "positive":
        print("Vary it: close and far from the mic, loud and quiet, fast and slow, a bit of room noise.")
        for i in range(a.n):
            input(f"[{i + 1}/{a.n}] press Enter, then say 'hey kalki' ...")
            clip = rec(2.0)
            if np.abs(clip).max() < 0.01:
                print("  too quiet, skipped (is the right microphone selected?)")
                continue
            sf.write(out / f"pos_real_{k}.wav", trim(clip), SR)
            k += 1
    else:
        chunks = int(a.minutes * 60 / 10)
        print(f"Recording {chunks} x 10 s of normal background and speech. Do NOT say 'hey kalki'.")
        input("press Enter to start ...")
        for i in range(chunks):
            sf.write(out / f"neg_real_{k}.wav", rec(10.0), SR)
            k += 1
            print(f"  {i + 1}/{chunks}", end="\r", flush=True)
    print(f"\nsaved to {out}. Next: python train.py --steps 6000")


if __name__ == "__main__":
    main()
