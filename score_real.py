"""Score the wake word model on your own held-out recordings, the way live.py would hear them.

Workflow (see README):
    python record_real.py positive -n 40
    python record_real.py negative --minutes 5
    python score_real.py split                       # set ~25% aside BEFORE training (moves files to real_test/)
    python score_real.py score --model models/kalki_int8.onnx --model out/kalki.onnx   # before vs after

Positives: each held-out "hey kalki" clip should fire the detector at least once.
Negatives: long clips of normal sound; every firing is a false alarm.
"""
import argparse
import json
import math
import random
import shutil
from pathlib import Path

import numpy as np
import onnxruntime as ort
import soundfile as sf
from scipy.signal import resample_poly

SR, WIN, HOP = 16000, 24000, 1280
HITS, COOLDOWN = 2, 1.5   # same trigger rule as live.py / detector.py
HERE = Path(__file__).parent


def load_wav(path):
    x, sr = sf.read(path, dtype="float32")
    if x.ndim > 1:
        x = x.mean(1)
    if sr != SR:
        g = math.gcd(sr, SR)
        x = resample_poly(x, SR // g, sr // g).astype(np.float32)
    return x


def probabilities(sess, x):
    """Probability the wake word just ended, every 80 ms (same windows as the live detector)."""
    x = np.concatenate([np.zeros(WIN - HOP, np.float32), x])
    n = (len(x) - WIN) // HOP + 1
    out = np.empty(n, np.float32)
    for i in range(n):
        w = x[i * HOP: i * HOP + WIN][None]
        out[i] = 1 / (1 + np.exp(-sess.run(None, {"audio": w})[0][0]))
    return out


def triggers(p, thr):
    """Apply the live trigger rule; returns the times (s) the detector would fire."""
    fired, streak, last = [], 0, -1e9
    for i, v in enumerate(p):
        streak = streak + 1 if v > thr else 0
        t = (i + 1) * HOP / SR
        if streak >= HITS and t - last > COOLDOWN:
            fired.append(t)
            last = t
    return fired


def wilson(k, n, z=1.96):
    if n == 0:
        return 0.0, 0.0
    p = k / n
    d = 1 + z * z / n
    c = (p + z * z / (2 * n)) / d
    h = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / d
    return max(0.0, c - h), min(1.0, c + h)


def clips(folder):
    files = sorted(Path(folder).glob("*.wav"))
    if not files:
        raise SystemExit(f"no .wav files in {folder}")
    return files


def padded(x, rng):
    """Recordings from record_real.py are trimmed tight; add a quiet lead-in and tail like a live stream."""
    quiet = lambda s: (rng.standard_normal(int(s * SR)) * 0.002).astype(np.float32)
    return np.concatenate([quiet(0.7), x, quiet(1.3)])


def score_model(model_path, pos_files, neg_files, thresholds):
    so = ort.SessionOptions(); so.intra_op_num_threads = 1
    sess = ort.InferenceSession(str(model_path), so, providers=["CPUExecutionProvider"])
    rng = np.random.default_rng(0)
    pos = {f: probabilities(sess, padded(load_wav(f), rng)) for f in pos_files}
    neg = {f: probabilities(sess, load_wav(f)) for f in neg_files}
    hours = sum(len(p) * HOP / SR for p in neg.values()) / 3600
    rows = []
    for thr in thresholds:
        caught = [f for f, p in pos.items() if triggers(p, thr)]
        false_by_file = {f.name: len(triggers(p, thr)) for f, p in neg.items()}
        n_false = sum(false_by_file.values())
        rows.append({"threshold": thr, "caught": len(caught), "positives": len(pos),
                     "missed_files": [f.name for f in pos if f not in caught],
                     "false_alarms": n_false, "negative_hours": hours,
                     "false_alarms_per_hour": n_false / hours if hours else float("nan"),
                     "false_alarm_files": {k: v for k, v in false_by_file.items() if v}})
    peaks = {f.name: float(p.max()) for f, p in pos.items()}
    return rows, peaks


def cmd_score(a):
    pos_files, neg_files = clips(Path(a.dir) / "positive"), clips(Path(a.dir) / "negative")
    models = a.model or [str(HERE / "models" / "kalki_int8.onnx")]
    thresholds = a.thresholds
    print(f"{len(pos_files)} held-out wake word clips, {len(neg_files)} negative clips\n")
    report = {}
    for m in models:
        rows, peaks = score_model(m, pos_files, neg_files, thresholds)
        report[m] = {"rows": rows, "positive_peak_probability": peaks}
        print(f"### {m}\n")
        print("| Threshold | Wake words caught | 95% interval | False alarms | False alarms / hour |")
        print("|----------:|------------------:|-------------:|-------------:|--------------------:|")
        for r in rows:
            lo, hi = wilson(r["caught"], r["positives"])
            print(f"| {r['threshold']} | {r['caught']}/{r['positives']} ({r['caught'] / r['positives']:.0%}) "
                  f"| {lo:.0%} to {hi:.0%} | {r['false_alarms']} in {r['negative_hours'] * 60:.1f} min "
                  f"| {r['false_alarms_per_hour']:.0f} |")
        main = next((r for r in rows if r["threshold"] == 0.8), rows[0])
        if main["missed_files"]:
            print(f"\nMissed at {main['threshold']}: " + ", ".join(main["missed_files"][:10]))
        if main["false_alarm_files"]:
            print(f"False alarms at {main['threshold']} (file: count): " +
                  ", ".join(f"{k}: {v}" for k, v in list(main["false_alarm_files"].items())[:10]))
        print()
    print("With about 10 positive clips each result is only +/- 20%, and a few minutes of negatives gives a very\n"
          "rough per-hour figure. Use these numbers to compare models on the same clips, not as a final grade.")
    if a.json:
        Path(a.json).write_text(json.dumps(report, indent=2))


def cmd_split(a):
    random.seed(a.seed)
    data, out = Path(a.data), Path(a.dir)
    for kind, n, pattern in (("positive", a.n_pos, "pos_real_*.wav"), ("negative", a.n_neg, "neg_real_*.wav")):
        files = sorted((data / kind).glob(pattern))
        if len(files) < 2:
            print(f"{kind}: need at least 2 recordings matching {pattern} in {data / kind}, found {len(files)}; skipped")
            continue
        n = min(n, len(files) - 1)  # always leave something to train on
        dest = out / kind
        dest.mkdir(parents=True, exist_ok=True)
        for f in random.sample(files, n):
            shutil.move(str(f), dest / f.name)
        print(f"{kind}: moved {n} of {len(files)} clips to {dest} (they will NOT be used for training)")


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest="cmd", required=True)
    s = sub.add_parser("split", help="move some of your recordings out of training into real_test/")
    s.add_argument("--data", default=str(HERE / "data"))
    s.add_argument("--dir", default=str(HERE / "real_test"))
    s.add_argument("--n-pos", type=int, default=10)
    s.add_argument("--n-neg", type=int, default=6)
    s.add_argument("--seed", type=int, default=0)
    s.set_defaults(fn=cmd_split)
    c = sub.add_parser("score", help="score one or more models on real_test/")
    c.add_argument("--dir", default=str(HERE / "real_test"))
    c.add_argument("--model", action="append", help="ONNX model; repeat to compare several")
    c.add_argument("--thresholds", type=float, nargs="+", default=[0.5, 0.7, 0.8, 0.9])
    c.add_argument("--json", help="also write the raw numbers here")
    c.set_defaults(fn=cmd_score)
    a = ap.parse_args()
    a.fn(a)


if __name__ == "__main__":
    main()
