import pytest

torch = pytest.importorskip("torch")

from model import WIN, WakeNet, spec_augment  # noqa: E402
from train import heldout_voices  # noqa: E402


def test_spec_augment_masks_bounded_bands():
    torch.manual_seed(0)
    y = spec_augment(torch.ones(32, 40, 150))
    z = y == 0
    assert 0 < z.float().mean() < 0.4 and z.all(2).sum(1).max() <= 6 and z.all(1).sum(1).max() <= 12


def test_spec_augment_only_in_training_mode_and_keeps_export_keys():
    m = WakeNet()
    assert not m.spec_augment and len(m.state_dict()) == 35          # same checkpoint layout as the shipped models
    m.spec_augment = True
    m.eval()
    x = torch.randn(2, WIN)
    a = m(x)
    m.spec_augment = False
    assert torch.equal(a, m(x))


def test_heldout_voices_ignore_second_tts_and_real(tmp_path):
    from pathlib import Path
    names = lambda vs: {Path(f"pos_{v}_0.wav"): 0 for v in vs}
    edge = [f"edge{i}" for i in range(16)]
    base = heldout_voices(names(edge))
    assert len(base) == 2
    assert heldout_voices(names(edge + ["real"] + [f"ppvctks{i}" for i in range(50)])) == base


def _sampler():
    import numpy as np
    from pathlib import Path
    from train import Sampler
    rng = np.random.default_rng(0)
    clip = lambda: (rng.standard_normal(12000) * 0.1).astype("float32")
    pos = {Path(f"pos_v{i}_0.wav"): clip() for i in range(3)}
    neg = {Path(f"{tag}_v{i}_0.wav"): clip() for i in range(3) for tag in ("sent", "hard")}   # Sampler needs hard negatives
    return Sampler(pos, neg, 1, augment=True)


def test_batches_in_workers_match_the_direct_path_and_differ_per_worker():
    import train
    for workers in (0, 2):
        it = train.batches(_sampler(), 8, "cpu", workers)
        seen = [next(it) for _ in range(4)]
        assert all(x.shape == (8, WIN) and y.shape == (8,) and x.dtype == torch.float32 for x, y in seen)
        assert all(float(y.sum()) == 4 for _, y in seen)                  # half positives, as before
        assert len({tuple(x[0, :5].tolist()) for x, _ in seen}) == 4      # workers do not repeat each other's batches
        del it


def test_default_workers_split_cores(monkeypatch):
    import os
    import sys
    import train
    monkeypatch.setattr(sys, "platform", "linux")
    for cores, cpu, gpu in ((1, 0, 0), (2, 0, 0), (4, 2, 3), (8, 4, 7), (64, 32, 8)):
        monkeypatch.setattr(os, "cpu_count", lambda c=cores: c)
        assert (train.default_workers("cpu"), train.default_workers("cuda")) == (cpu, gpu)
    monkeypatch.setattr(sys, "platform", "darwin")
    assert train.default_workers("cpu") == 0
