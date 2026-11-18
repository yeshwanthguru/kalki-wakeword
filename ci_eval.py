"""Score the committed ONNX models on the held-out voices and compare with docs/baseline.json.

    python ci_eval.py                  # print + write ci_report.json / ci_report.md
    python ci_eval.py --update-baseline
Exit code 1 if AUC drops more than --auc-tol or stream false alarms/hour rise more than --fa-tol.
Needs data/ (python gen_tts.py) and the models in models/.
"""
import argparse
import json
import sys

import numpy as np

from benchmark import HERE, SR, WIN, build_stream, roc_metrics, splits, trigger_stats
from quantize import probs
from train import Sampler

MODELS = {"fp32": HERE / "models" / "kalki_fp32.onnx", "int8": HERE / "models" / "kalki_int8.onnx"}
BASELINE = HERE / "docs" / "baseline.json"


def evaluate():
    _, _, vp, vn = splits()
    x, y = Sampler(vp, vn, 123).batch(4000, "cpu")
    x, y = x.numpy(), y.numpy() > 0.5
    stream, ends = build_stream(vp, vn)
    starts = np.arange(0, len(stream) - WIN, SR // 12)
    windows = np.stack([stream[i:i + WIN] for i in starts])
    out = {}
    for name, path in MODELS.items():
        px = probs(path, x)
        r, _ = roc_metrics(px[y], px[~y])
        p = probs(path, windows)
        r["stream"] = {}
        for thr in (0.5, 0.8, 0.9):
            det, fa = trigger_stats(p, starts, len(stream), ends, thr=thr)
            r["stream"][str(thr)] = {"detected": det, "false_alarms_per_hour": fa}
        out[name] = r
    return out


def report(cur, base, auc_tol, fa_tol):
    lines, fail = ["| model | AUC | baseline AUC | recall@0.5 | FPR@0.5 | caught@0.8 | false alarms/h @0.8 | baseline FA/h |",
                   "|---|---:|---:|---:|---:|---:|---:|---:|"], []
    for name, r in cur.items():
        b = (base or {}).get(name)
        s = r["stream"]["0.8"]
        lines.append(f"| {name} | {r['auc']:.4f} | {b['auc']:.4f} |" if b else f"| {name} | {r['auc']:.4f} | n/a |")
        lines[-1] += (f" {r['recall@0.5']:.3f} | {r['fpr@0.5']:.3f} | {s['detected']:.0%} | {s['false_alarms_per_hour']:.0f} | "
                      + (f"{b['stream']['0.8']['false_alarms_per_hour']:.0f} |" if b else "n/a |"))
        if b and r["auc"] < b["auc"] - auc_tol:
            fail.append(f"{name}: AUC {r['auc']:.4f} < baseline {b['auc']:.4f} - {auc_tol}")
        if b and s["false_alarms_per_hour"] > b["stream"]["0.8"]["false_alarms_per_hour"] + fa_tol:
            fail.append(f"{name}: false alarms/h {s['false_alarms_per_hour']:.0f} > baseline "
                        f"{b['stream']['0.8']['false_alarms_per_hour']:.0f} + {fa_tol}")
    return "\n".join(lines), fail


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--auc-tol", type=float, default=0.005)
    ap.add_argument("--fa-tol", type=float, default=8.0, help="allowed rise in false alarms per hour at thr 0.8")
    ap.add_argument("--update-baseline", action="store_true")
    a = ap.parse_args()
    cur = evaluate()
    if a.update_baseline:
        BASELINE.write_text(json.dumps(cur, indent=2)); print("wrote", BASELINE); sys.exit(0)
    base = json.loads(BASELINE.read_text()) if BASELINE.exists() else None
    table, fail = report(cur, base, a.auc_tol, a.fa_tol)
    md = table + ("\n\n**Regression:**\n" + "\n".join(f"- {f}" for f in fail) if fail else
                  "\n\nNo regression vs `docs/baseline.json`." if base else "\n\nNo baseline committed yet.")
    (HERE / "ci_report.json").write_text(json.dumps(cur, indent=2))
    (HERE / "ci_report.md").write_text(md)
    print(md)
    sys.exit(1 if fail else 0)
