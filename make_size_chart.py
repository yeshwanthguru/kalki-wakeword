"""Redraw docs/model_sizes.png and print the model-files table for the README, from the frozen evaluation set.

    python eval_set.py fetch            # once
    python make_size_chart.py [--out docs/model_sizes.png]
Scores the four model files on eval-set-v2 (same 4,000 test windows for all of them, plus the 60-minute stream at the default
trigger) and measures one-thread CPU time per window on this machine, so the numbers in the README are reproducible, not typed.
"""
import argparse
import os
import time
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import onnxruntime as ort  # noqa: E402

import eval_set  # noqa: E402
from benchmark import SR, WIN, build_stream, roc_metrics, trigger_stats  # noqa: E402
from quantize import probs  # noqa: E402
from train import Sampler  # noqa: E402

HERE = Path(__file__).parent
FILES = [("fp32", "fp32\n(original)", "#9aa0a8"), ("fp16", "fp16\n(half precision)", "#5fb7a0"),
         ("int8", "int8\n(8-bit weights)", "#0f7b5f"), ("int4", "int4\n(4-bit weights)", "#0a5c46")]


def ms_per_window(path, window):
    so = ort.SessionOptions(); so.intra_op_num_threads = 1
    sess = ort.InferenceSession(str(path), so, providers=["CPUExecutionProvider"])
    for _ in range(30):
        sess.run(None, {"audio": window})
    t = time.perf_counter()
    for _ in range(300):
        sess.run(None, {"audio": window})
    return (time.perf_counter() - t) / 300 * 1000


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--out", default=str(HERE / "docs" / "model_sizes.png"))
    a = ap.parse_args()
    if not eval_set.exists():
        raise SystemExit("no eval_set/: run `python eval_set.py fetch` first")
    vp, vn = eval_set.load_set()
    x, y = Sampler(vp, vn, 123).batch(4000, "cpu")
    x, y = x.numpy(), y.numpy() > 0.5
    stream, ends = build_stream(vp, vn, minutes=eval_set.stream_minutes())
    starts = np.arange(0, len(stream) - WIN, SR // 12)
    windows = np.stack([stream[i:i + WIN] for i in starts])
    rows = {}
    for name, _, _ in FILES:
        path = HERE / "models" / f"kalki_{name}.onnx"
        p = probs(path, x)
        r, _ = roc_metrics(p[y], p[~y])
        caught, fa = trigger_stats(probs(path, windows), starts, len(stream), ends, thr=0.8, hits=2)
        rows[name] = dict(kb=os.path.getsize(path) / 1024, auc=r["auc"], recall=r["recall@0.5"], fpr=r["fpr@0.5"], caught=caught, fa=fa,
                          ms=ms_per_window(path, windows[100][None]))
    fp32kb = rows["fp32"]["kb"]

    fig, ax = plt.subplots(1, 2, figsize=(11, 4.4))
    names = [n for n, _, _ in FILES]
    colors = [c for _, _, c in FILES]
    labels = [l for _, l, _ in FILES]
    sizes = [rows[n]["kb"] for n in names]
    ax[0].bar(labels, sizes, color=colors, width=0.62)
    for i, n in enumerate(names):
        t = f"{sizes[i]:.0f} KB" + ("" if n == "fp32" else f"\n{100 * (1 - sizes[i] / fp32kb):.0f}% smaller")
        ax[0].text(i, sizes[i] + 20, t, ha="center", va="bottom", fontweight="bold", fontsize=10)
    ax[0].set_ylabel("file size (KB)"); ax[0].set_ylim(0, max(sizes) * 1.25); ax[0].set_title("Size on disk", loc="left", fontweight="bold")
    rec = [100 * rows[n]["recall"] for n in names]
    ax[1].bar(labels, rec, color=colors, width=0.62)
    for i, v in enumerate(rec):
        ax[1].text(i, v + 1, f"{v:.1f}%", ha="center", va="bottom", fontweight="bold", fontsize=10)
    ax[1].set_ylim(0, 110); ax[1].set_ylabel("wake words caught (%)"); ax[1].set_title("Accuracy is about the same", loc="left", fontweight="bold")
    for axis in ax:
        axis.spines[["top", "right"]].set_visible(False); axis.tick_params(axis="x", labelsize=9)
    fig.text(0.5, 0.005, "Held-out synthetic voices (eval-set-v2), threshold 0.5, same 4,000-window test set for all four files",
             ha="center", fontsize=9, color="#666")
    fig.tight_layout(rect=(0, 0.03, 1, 1)); fig.savefig(a.out, dpi=130); print("wrote", a.out)

    print("\n| File | Precision | Size | Wake words caught @0.5 | AUC | Caught / false alarms per hour (0.8 x 2) | CPU / window |")
    print("|------|-----------|-----:|-----------------------:|----:|------------------------------------------:|-------------:|")
    prec = {"fp32": "32-bit", "fp16": "16-bit", "int8": "8-bit", "int4": "4-bit, front end computed at load"}
    for n in ["int4", "int8", "fp16", "fp32"]:
        r = rows[n]
        pct = "(original)" if n == "fp32" else f"(-{100 * (1 - r['kb'] / fp32kb):.0f}%)"
        print(f"| [`kalki_{n}.onnx`](models/kalki_{n}.onnx) | {prec[n]} | {'**' if n == 'int4' else ''}{r['kb']:.0f} KB{'**' if n == 'int4' else ''} {pct} | "
              f"{100 * r['recall']:.1f}% | {r['auc']:.4f} | {100 * r['caught']:.1f}% / {r['fa']:.0f} | ~{r['ms']:.1f} ms |")


if __name__ == "__main__":
    main()
