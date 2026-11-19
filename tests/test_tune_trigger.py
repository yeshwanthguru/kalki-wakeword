import pytest

pytest.importorskip("torch")

import numpy as np  # noqa: E402

import tune_trigger as tt  # noqa: E402
from benchmark import SR, WIN  # noqa: E402


def stream_probs():
    """Windows every 80 ms for 40 s: a real wake word (3 windows high) near 10 s and 30 s, one single-window blip at 20 s."""
    hop = SR // 12
    starts = np.arange(0, 40 * SR, hop)
    p = np.full(len(starts), 0.05, np.float32)
    for sec in (10, 30):
        k = int(sec * SR / hop)
        p[k:k + 3] = 0.95
    p[int(20 * SR / hop)] = 0.95
    ends = [10 * SR + 2 * hop + WIN, 30 * SR + 2 * hop + WIN]
    return p, starts, 40 * SR, ends


def test_more_hits_remove_the_blip_but_keep_the_real_wake_words():
    p, starts, n, ends = stream_probs()
    grid = tt.sweep(p, starts, n, ends, thresholds=(0.8,), hits=(1, 2, 3, 4))
    assert grid[(1, 0.8)][1] > 0 and grid[(2, 0.8)][1] == 0 and grid[(3, 0.8)][1] == 0     # blip needs 1 hit only
    assert grid[(3, 0.8)][0] == 1.0 and grid[(4, 0.8)][0] == 0.0                           # 4 hits miss a 3-window word


def test_recommend_trades_recall_loss_for_fewer_false_alarms():
    grid = {(2, 0.8): (0.97, 27.0), (3, 0.8): (0.97, 22.0), (3, 0.9): (0.95, 13.0), (4, 0.9): (0.92, 4.0), (1, 0.5): (0.98, 81.0)}
    assert tt.recommend(grid, 0.01) == (3, 0.8)       # within 1 point of the best 98%: 97% / 22
    assert tt.recommend(grid, 0.03) == (3, 0.9)       # 95% allowed: 13 false alarms
    assert tt.recommend(grid, 0.0) == (1, 0.5)        # no loss allowed: the best recall


def test_table_marks_default_and_recommendation():
    grid = {(h, t): (0.9, 5.0) for h in tt.HITS for t in tt.THRESHOLDS}
    md = tt.table(grid, pick=(3, 0.9))
    assert md.count("(default)") == 1 and md.count("(recommended)") == 1 and md.splitlines()[0].startswith("| hits |")
