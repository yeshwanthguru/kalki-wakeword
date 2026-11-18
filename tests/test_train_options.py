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
