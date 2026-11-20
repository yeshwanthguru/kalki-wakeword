"""shrink_model.py: the committed 36 KB model is exactly what the script builds, and it behaves like the fp32 model."""
from pathlib import Path

import numpy as np
import pytest
import soundfile as sf

onnx = pytest.importorskip("onnx")
import onnxruntime as ort  # noqa: E402

import shrink_model  # noqa: E402

ROOT = Path(__file__).resolve().parent.parent
CLIPS = ROOT / "docs" / "clips"


def test_committed_model_is_what_the_script_builds(tmp_path):
    out = tmp_path / "rebuilt.onnx"
    size = shrink_model.build_int4(ROOT / "models" / "kalki_fp32.onnx", out)
    assert out.read_bytes() == (ROOT / "models" / "kalki_int4.onnx").read_bytes()    # deterministic and in sync with fp32
    assert size < 50_000


def test_stored_front_end_tables_are_gone_and_weights_are_4_bit():
    m = onnx.load(str(ROOT / "models" / "kalki_int4.onnx"))
    names = {i.name for i in m.graph.initializer}
    assert not names & set(shrink_model.FRONT_END)
    assert sum(i.data_type == onnx.TensorProto.INT4 for i in m.graph.initializer) == 5      # 3 conv layers + 2 GRU matrices
    assert max(op.version for op in m.opset_import if op.domain in ("", "ai.onnx")) == shrink_model.OPSET


def test_logits_stay_close_to_fp32_and_decisions_agree():
    fp32 = ort.InferenceSession(str(ROOT / "models" / "kalki_fp32.onnx"), providers=["CPUExecutionProvider"])
    int4 = ort.InferenceSession(str(ROOT / "models" / "kalki_int4.onnx"), providers=["CPUExecutionProvider"])
    diffs, agree, total = [], 0, 0
    for name in ("hey-kalki-1", "hey-kalki-2", "hey-kalki-3", "hey-google", "sentence"):
        x, _ = sf.read(CLIPS / f"{name}.wav", dtype="float32")
        x = np.concatenate([np.zeros(24000, np.float32), x])
        for s in range(0, len(x) - 24000, 1280):
            w = x[s:s + 24000][None]
            a, b = fp32.run(None, {"audio": w})[0][0], int4.run(None, {"audio": w})[0][0]
            diffs.append(abs(float(a) - float(b)))
            agree += (a > 1.386) == (b > 1.386)          # logit of probability 0.8
            total += 1
    assert np.mean(diffs) < 0.5 and agree / total > 0.98, (np.mean(diffs), agree / total)


def test_constants_match_model_py():
    torch = pytest.importorskip("torch")
    import model
    assert (shrink_model.SR, shrink_model.N_FFT, shrink_model.N_MELS) == (model.SR, model.N_FFT, model.N_MELS)
    # the mel triangles computed from three edges per filter are the stored filter bank
    lo, c, hi = shrink_model._mel_edges()
    hz = np.arange(shrink_model.N_FFT // 2 + 1, dtype=np.float32) * np.float32(shrink_model.SR / 2 / (shrink_model.N_FFT // 2))
    fb = np.maximum(np.minimum((hz - lo) / (c - lo), (hi - hz) / (hi - c)), 0)
    assert np.abs(fb - model.mel_filterbank().numpy()).max() < 1e-5
