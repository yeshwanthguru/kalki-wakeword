"""Score two checkpoints (default: first version vs current) on the same fixed held-out test set and stream.

    python compare_versions.py [--a out/kalki_v1.pt] [--b out/kalki.pt]
"""
import argparse
import json

import numpy as np
import torch

from benchmark import DOCS, HERE, SR, build_stream, roc_metrics, scores, splits, stream_eval
from model import WakeNet
from train import Sampler


def load(path):
    m = WakeNet(); m.load_state_dict(torch.load(path, map_location="cpu")); return m.eval()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--a", default=str(HERE / "out" / "kalki_v1.pt"))
    ap.add_argument("--b", default=str(HERE / "out" / "kalki.pt"))
    a = ap.parse_args()
    _, _, vp, vn = splits()
    x, y = Sampler(vp, vn, 123).batch(4000, "cpu")
    y = y.numpy() > 0.5
    stream, ends = build_stream(vp, vn)
    out = {}
    for name, path in (("v1", a.a), ("v2", a.b)):
        m = load(path)
        p = scores(m, x)
        r, _ = roc_metrics(p[y], p[~y])
        for thr in (0.5, 0.8, 0.9):
            r[f"stream_detect@{thr}"], r[f"stream_fa_per_hour@{thr}"] = stream_eval(m, stream, ends, thr=thr)
        out[name] = r
    keys = list(out["v1"])
    print(f"{'metric':<26}{'v1':>10}{'v2':>10}")
    for k in keys:
        print(f"{k:<26}{out['v1'][k]:>10.4f}{out['v2'][k]:>10.4f}")
    DOCS.mkdir(exist_ok=True)
    (DOCS / "v1_vs_v2.json").write_text(json.dumps(out, indent=2))


if __name__ == "__main__":
    main()
