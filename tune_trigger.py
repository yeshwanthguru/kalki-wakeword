"""Pick the trigger settings (probability threshold x number of consecutive hits) for a model on the frozen evaluation set.

    python tune_trigger.py [--model models/kalki_fp32.onnx] [--max-recall-loss 0.01] [--markdown out.md] [--json out.json]
Scores the held-out stream once, then replays the detector's trigger rule (what detector.py / live.py / the browser demo do:
fire when `hits` windows in a row exceed `threshold`, then wait out the cooldown) for every combination. Prints wake words
caught and false alarms per hour for each, marks the current default (0.8 x 2), and recommends the setting with the fewest
false alarms among those that lose at most --max-recall-loss of the best recall. Needs eval_set/ (python eval_set.py fetch).
Each extra hit adds one 80 ms window of delay. The result is for synthetic held-out voices: confirm on real recordings
(score_real.py) before changing a default.
"""
import argparse
import json
from pathlib import Path

import numpy as np

import eval_set
from benchmark import SR, WIN, build_stream, trigger_stats
from quantize import probs

HERE = Path(__file__).parent
THRESHOLDS = (0.5, 0.6, 0.7, 0.8, 0.9, 0.95)
HITS = (1, 2, 3, 4)
DEFAULT = (2, 0.8)   # detector.py: hits=2, threshold=0.8


def sweep(p, starts, n_samples, ends, thresholds=THRESHOLDS, hits=HITS):
    """{(hits, threshold): (fraction of wake words caught, false alarms per hour)} from per-window probabilities."""
    return {(h, t): trigger_stats(p, starts, n_samples, ends, thr=t, hits=h) for h in hits for t in thresholds}


def recommend(grid, max_recall_loss=0.01):
    """The (hits, threshold) with the fewest false alarms among settings within max_recall_loss of the best recall.
    Ties go to the higher recall, then fewer hits (less delay)."""
    best = max(d for d, _ in grid.values())
    ok = [(fa, -d, h, t) for (h, t), (d, fa) in grid.items() if d >= best - max_recall_loss - 1e-9]
    fa, nd, h, t = min(ok)
    return (h, t)


def table(grid, thresholds=THRESHOLDS, hits=HITS, mark=DEFAULT, pick=None):
    lines = ["| hits | " + " | ".join(f"thr {t}" for t in thresholds) + " |", "|---:|" + "---:|" * len(thresholds)]
    for h in hits:
        cells = []
        for t in thresholds:
            d, fa = grid[(h, t)]
            cell = f"{d:.0%} / {fa:.0f}"
            if (h, t) == pick:
                cell = f"**{cell}** (recommended)"
            elif (h, t) == mark:
                cell += " (default)"
            cells.append(cell)
        lines.append(f"| {h} | " + " | ".join(cells) + " |")
    return "\n".join(lines)


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--model", default=str(HERE / "models" / "kalki_fp32.onnx"))
    ap.add_argument("--max-recall-loss", type=float, default=0.01)
    ap.add_argument("--markdown")
    ap.add_argument("--json")
    a = ap.parse_args()
    if not eval_set.exists():
        raise SystemExit("no eval_set/: run `python eval_set.py fetch` first")
    vp, vn = eval_set.load_set()
    stream, ends = build_stream(vp, vn, minutes=eval_set.stream_minutes())
    starts = np.arange(0, len(stream) - WIN, SR // 12)
    p = probs(a.model, np.stack([stream[i:i + WIN] for i in starts]))
    grid = sweep(p, starts, len(stream), ends)
    pick = recommend(grid, a.max_recall_loss)
    d0, fa0 = grid[DEFAULT]
    d1, fa1 = grid[pick]
    md = (f"Trigger settings for `{Path(a.model).name}` on eval set `{eval_set.version()}` "
          f"({len(stream) / SR / 60:.0f} min, {len(ends)} wake words). Each cell: wake words caught / false alarms per hour.\n\n"
          + table(grid, pick=pick) + "\n\n"
          + f"Default (hits {DEFAULT[0]}, threshold {DEFAULT[1]}): {d0:.1%} caught, {fa0:.0f} false alarms/h. "
          + f"Recommended within {a.max_recall_loss:.1%} of the best recall: hits {pick[0]}, threshold {pick[1]}: "
          + f"{d1:.1%} caught, {fa1:.0f} false alarms/h (+{(pick[0] - 1) * 80} ms delay vs 1 hit).")
    print(md)
    if a.markdown:
        Path(a.markdown).write_text(md)
    if a.json:
        Path(a.json).write_text(json.dumps({"eval_set": eval_set.version(), "minutes": len(stream) / SR / 60, "wake_words": len(ends),
                                            "recommended": {"hits": pick[0], "threshold": pick[1]},
                                            "grid": {f"{h}@{t}": {"caught": d, "false_alarms_per_hour": fa} for (h, t), (d, fa) in grid.items()}},
                                           indent=1))


if __name__ == "__main__":
    main()
