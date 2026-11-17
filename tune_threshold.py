"""How the trigger threshold trades missed wake words for false alarms on the simulated 15-min stream.

    python tune_threshold.py [--ckpt out/kalki.pt]
Prints a markdown table and writes docs/threshold_sweep.json.
"""
import argparse
import json

import torch

from benchmark import DOCS, HERE, SR, build_stream, splits, stream_eval
from model import WakeNet


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", default=str(HERE / "out" / "kalki.pt"))
    a = ap.parse_args()
    m = WakeNet(); m.load_state_dict(torch.load(a.ckpt, map_location="cpu")); m.eval()
    _, _, vp, vn = splits()
    stream, ends = build_stream(vp, vn)
    rows = []
    print(f"stream: {len(stream) / SR / 60:.0f} min, {len(ends)} wake words (held-out voices)")
    print("| Threshold | Wake words caught | False alarms per hour |\n|---:|---:|---:|")
    for thr in (0.5, 0.6, 0.7, 0.8, 0.9, 0.95):
        det, fa = stream_eval(m, stream, ends, thr=thr)
        rows.append({"threshold": thr, "detected": det, "false_alarms_per_hour": fa})
        print(f"| {thr} | {det:.0%} | {fa:.0f} |")
    DOCS.mkdir(exist_ok=True)
    (DOCS / "threshold_sweep.json").write_text(json.dumps(rows, indent=2))


if __name__ == "__main__":
    main()
