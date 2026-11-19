"""Summarize several retrain runs: mean and range of every score per group, and which groups differ from a reference group.

    python summarize_runs.py control=run4/ci_report.json control=run5/ci_report.json control=run6/ci_report.json \\
                             specaug=run7/ci_report.json --reference control [--model int8]
Each argument is label=path to the ci_report.json that a retrain run uploads (artifact 'retrain-report'). Runs with the same
label form a group. With --reference, a metric is marked when a group's mean lies outside the reference group's min..max
widened by --margin of its range, i.e. only clearly larger-than-noise differences are called out. With few identical runs the
range is a rough noise estimate: three runs cannot prove that something is NOT different.
"""
import argparse
import json
import statistics
import sys
from collections import defaultdict
from pathlib import Path

METRICS = [
    ("AUC", lambda r: r["auc"], "{:.4f}"),
    ("recall@0.5", lambda r: r["recall@0.5"], "{:.3f}"),
    ("FPR@0.5", lambda r: r["fpr@0.5"], "{:.3f}"),
    ("caught@0.8", lambda r: r["stream"]["0.8"]["detected"], "{:.1%}"),
    ("false alarms/h @0.8", lambda r: r["stream"]["0.8"]["false_alarms_per_hour"], "{:.0f}"),
]


def load_groups(items, model="int8"):
    groups, sets = defaultdict(list), set()
    for item in items:
        label, _, path = item.partition("=")
        if not path:
            raise SystemExit(f"expected label=path, got {item!r}")
        d = json.loads(Path(path).read_text())
        sets.add(d.get("eval_set", "regenerated"))
        groups[label].append(d[model])
    return groups, sets


def summarize(groups, reference=None, margin=0.5):
    """Markdown table: one row per group; 'n' runs; mean [min-max] per metric; * where clearly outside the reference range."""
    ref = groups.get(reference) if reference else None
    lines = ["| group | n | " + " | ".join(m[0] for m in METRICS) + " |", "|---|---:|" + "---:|" * len(METRICS)]
    flagged = []
    for label, runs in groups.items():
        cells = []
        for name, get, fmt in METRICS:
            vals = [get(r) for r in runs]
            mean = statistics.fmean(vals)
            cell = fmt.format(mean) if len(vals) == 1 else f"{fmt.format(mean)} [{fmt.format(min(vals))}-{fmt.format(max(vals))}]"
            if ref and label != reference:
                rv = [get(r) for r in ref]
                lo, hi, width = min(rv), max(rv), max(rv) - min(rv)
                if mean < lo - margin * width or mean > hi + margin * width:
                    cell += " *"
                    flagged.append((label, name))
            cells.append(cell)
        lines.append(f"| {label} | {len(runs)} | " + " | ".join(cells) + " |")
    if ref:
        lines += ["", f"`*` = group mean outside the `{reference}` range (min-max, widened by {margin:.0%} of its width)."
                  + (f" Flagged: {', '.join(f'{l}: {n}' for l, n in flagged)}." if flagged else " Nothing flagged: no clear difference.")]
        if len(ref) < 3:
            lines.append(f"Only {len(ref)} reference run(s): the range says little about noise.")
    return "\n".join(lines)


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("runs", nargs="+", help="label=path/to/ci_report.json (repeat a label for repeated runs)")
    ap.add_argument("--reference", help="label of the group to compare the others against")
    ap.add_argument("--model", default="int8", choices=["fp32", "int8"])
    ap.add_argument("--margin", type=float, default=0.5)
    a = ap.parse_args(argv)
    groups, sets = load_groups(a.runs, a.model)
    if a.reference and a.reference not in groups:
        raise SystemExit(f"reference group {a.reference!r} not among {sorted(groups)}")
    print(summarize(groups, a.reference, a.margin))
    if len(sets) > 1:
        print(f"\nWARNING: the runs were scored on different evaluation sets ({', '.join(sorted(sets))}); do not compare them.",
              file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
