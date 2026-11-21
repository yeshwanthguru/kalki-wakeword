import numpy as np
import onnxruntime as ort
import pytest

torch = pytest.importorskip("torch")   # the detector CI job installs no torch
import torch.nn.functional as F  # noqa: E402

import snn  # noqa: E402
from model import WIN  # noqa: E402


def _audio(n, seed=0):
    g = torch.Generator().manual_seed(seed)
    return torch.randn(n, WIN, generator=g) * 0.1


def test_forward_shape_and_params():
    m = snn.SpikingWakeNet().eval()
    assert m(_audio(3)).shape == (3,)
    assert 30_000 < sum(p.numel() for p in m.parameters()) < 70_000


def test_spikes_are_binary_in_eval():
    v = torch.randn(1000)
    s = snn.spike(v, training=False)
    assert set(s.unique().tolist()) <= {0.0, 1.0}
    assert torch.equal(s, (v >= 0).float())


def test_surrogate_gradient_flows_to_all_weights():
    m = snn.SpikingWakeNet().train()
    loss = F.binary_cross_entropy_with_logits(m(_audio(8) * 5), torch.ones(8))
    loss.backward()
    for name in ("fc1.weight", "fc2.weight", "rec.weight", "out.weight", "raw_beta1"):
        g = dict(m.named_parameters())[name].grad
        assert g is not None and torch.isfinite(g).all() and g.abs().sum() > 0, name


def test_can_fit_a_tiny_task():
    # loud tone bursts vs quiet noise: the loss must fall, so the surrogate gradient really trains the network
    torch.manual_seed(0)
    m = snn.SpikingWakeNet().train()
    opt = torch.optim.Adam(m.parameters(), lr=5e-3)
    t = torch.arange(WIN) / 16000
    x = torch.randn(16, WIN) * 0.05
    x[:8] += 0.5 * torch.sin(2 * np.pi * 1000 * t)[None] * (t > 1.0)[None]
    y = torch.cat([torch.ones(8), torch.zeros(8)])
    first = None
    for _ in range(40):
        loss = F.binary_cross_entropy_with_logits(m(x), y)
        first = first or loss.item()
        opt.zero_grad(); loss.backward(); opt.step()
    assert loss.item() < first * 0.7


def test_op_counts():
    m = snn.SpikingWakeNet()
    c = snn.op_counts(m, _audio(4))
    assert c["steps"] == 74
    assert c["spikes_per_window"]["layer1"] >= 0
    max_adds = c["dense_synapses_if_not_event_driven"]
    assert 0 <= c["synaptic_adds"] <= max_adds
    assert c["dense_macs"] > 0
    assert not m.record


def test_onnx_export_matches_torch(tmp_path):
    m = snn.SpikingWakeNet().eval()
    path = tmp_path / "snn.onnx"
    snn.export_onnx(m, path)
    sess = ort.InferenceSession(str(path), providers=["CPUExecutionProvider"])
    got, want = [], []
    for seed in (3, 4):
        x = _audio(1, seed=seed)
        with torch.no_grad():
            want.append(m(x).numpy())
        got.append(sess.run(None, {"audio": x.numpy()})[0])
    got, want = np.concatenate(got), np.concatenate(want)
    # a spike that sits exactly on the threshold could flip, so allow a little slack
    assert np.allclose(got, want, atol=1e-2), (got, want)
