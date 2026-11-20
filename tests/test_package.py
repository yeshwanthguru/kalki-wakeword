import re
import wave
from pathlib import Path

import numpy as np
import pytest

import detector
import kalki_wakeword
from kalki_wakeword import __main__ as cli

ROOT = Path(__file__).resolve().parent.parent
CLIPS = ROOT / "docs" / "clips"


def test_bundled_model_is_the_shipped_model():
    """The package carries its own copy of the int8 model (wheels cannot reach outside the package); keep it identical."""
    assert kalki_wakeword.DEFAULT_MODEL.read_bytes() == (ROOT / "models" / "kalki_int8.onnx").read_bytes()


def test_root_detector_shim_exports_the_package_objects():
    for name in ("WakeWordDetector", "Detection", "DEFAULT_MODEL", "SR", "WIN", "HOP"):
        assert getattr(detector, name) is getattr(kalki_wakeword, name)


def test_version_is_a_release_number_and_matches_pyproject():
    assert re.fullmatch(r"\d+\.\d+\.\d+", kalki_wakeword.__version__)
    py = (ROOT / "pyproject.toml").read_text()
    assert 'attr = "kalki_wakeword.__version__"' in py and "dynamic = [\"version\"]" in py


def test_cli_detects_and_reports(capsys):
    assert cli.main([str(CLIPS / "hey-kalki-1.wav")]) == 0
    assert "1 detection(s)" in capsys.readouterr().out
    assert cli.main([str(CLIPS / "hey-google.wav")]) == 1
    assert "0 detection(s)" in capsys.readouterr().out


def write_wav(path, x, rate=16000, channels=1):
    pcm = (np.clip(x, -1, 1) * 32767).astype("<i2")
    with wave.open(str(path), "wb") as w:
        w.setnchannels(channels); w.setsampwidth(2); w.setframerate(rate)
        w.writeframes(pcm.tobytes())


def test_cli_rejects_wrong_rate_and_averages_channels(tmp_path):
    write_wav(tmp_path / "slow.wav", np.zeros(8000), rate=8000)
    with pytest.raises(SystemExit) as e:
        cli.read_wav(tmp_path / "slow.wav")
    assert "16000 Hz" in str(e.value)
    stereo = np.stack([np.full(1600, 0.5), np.full(1600, -0.5)], 1)
    write_wav(tmp_path / "st.wav", stereo.ravel(), channels=2)
    mono = cli.read_wav(tmp_path / "st.wav")
    assert mono.shape == (1600,) and abs(mono).max() < 1e-3          # opposite channels cancel


def test_javascript_package_has_the_same_model_and_version():
    import json
    assert (ROOT / "js" / "models" / "kalki_int8.onnx").read_bytes() == (ROOT / "models" / "kalki_int8.onnx").read_bytes()
    assert json.loads((ROOT / "js" / "package.json").read_text())["version"] == kalki_wakeword.__version__


def test_sensitivity_presets_and_overrides():
    from kalki_wakeword import SENSITIVITY, WakeWordDetector
    assert SENSITIVITY == {"sensitive": (0.6, 2), "balanced": (0.8, 2), "strict": (0.9, 3)}
    d = WakeWordDetector()                                   # unchanged default: 0.8 and 2
    assert (d.threshold, d.hits) == (0.8, 2)
    s = WakeWordDetector(sensitivity="strict")
    assert (s.threshold, s.hits) == (0.9, 3)
    o = WakeWordDetector(sensitivity="strict", threshold=0.5)   # an explicit value overrides only that part
    assert (o.threshold, o.hits) == (0.5, 3)
    assert (WakeWordDetector(threshold=0.7, hits=4).threshold, WakeWordDetector(threshold=0.7, hits=4).hits) == (0.7, 4)
    with pytest.raises(ValueError, match="sensitivity"):
        WakeWordDetector(sensitivity="paranoid")


@pytest.mark.parametrize("sensitivity", ["sensitive", "balanced", "strict"])
def test_every_preset_still_detects_the_demo_clips(sensitivity):
    import soundfile as sf
    from kalki_wakeword import WakeWordDetector
    for name, expected in [("hey-kalki-1", 1), ("hey-kalki-2", 1), ("hey-kalki-3", 1), ("hey-google", 0), ("sentence", 0)]:
        x, _ = sf.read(CLIPS / f"{name}.wav", dtype="float32")
        det = WakeWordDetector(sensitivity=sensitivity)
        assert len(det.push(x) + det.push(np.zeros(4000, np.float32))) == expected, (sensitivity, name)


def test_cli_sensitivity_flag(capsys):
    assert cli.main([str(CLIPS / "hey-kalki-1.wav"), "--sensitivity", "strict"]) == 0
    assert cli.main([str(CLIPS / "hey-kalki-1.wav"), "--hits", "200"]) == 1          # an override that can never be met


def test_javascript_presets_match_python():
    import json
    import re
    from kalki_wakeword import SENSITIVITY
    js = (ROOT / "js" / "index.js").read_text()
    body = re.search(r"export const SENSITIVITY = \{(.*?)\};", js, re.S).group(1)
    parsed = {k: tuple(json.loads(v)) for k, v in re.findall(r"(\w+):\s*(\[[^\]]*\])", body)}
    assert parsed == {k: tuple(v) for k, v in SENSITIVITY.items()}


def test_all_model_copies_are_identical_and_the_small_one_is_small():
    for name in ("kalki_int4.onnx", "kalki_int8.onnx"):
        reference = (ROOT / "models" / name).read_bytes()
        for copy in (ROOT / "kalki_wakeword" / "models", ROOT / "js" / "models", ROOT / "docs"):
            assert (copy / name).read_bytes() == reference, f"{copy.name}/{name} differs from models/{name}"
    assert (ROOT / "models" / "kalki_int4.onnx").stat().st_size < 50_000          # the 50 KB goal


def test_default_model_is_the_36kb_one_and_falls_back_to_int8(monkeypatch, recwarn):
    import onnxruntime as ort
    from kalki_wakeword import DEFAULT_MODEL, INT4_MODEL, WakeWordDetector
    assert WakeWordDetector().model_path == INT4_MODEL
    assert WakeWordDetector(DEFAULT_MODEL).model_path == DEFAULT_MODEL            # an explicit path is used as given
    real = ort.InferenceSession

    def old_runtime(path, *a, **k):          # an onnxruntime that cannot read 4-bit weights
        if str(path).endswith("kalki_int4.onnx"):
            raise ort.capi.onnxruntime_pybind11_state.Fail("[ONNXRuntimeError] : 1 : FAIL : unsupported type int4")
        return real(path, *a, **k)
    monkeypatch.setattr(ort, "InferenceSession", old_runtime)
    with pytest.warns(RuntimeWarning, match="int8"):
        det = WakeWordDetector()
    assert det.model_path == DEFAULT_MODEL
    x, _ = __import__("soundfile").read(CLIPS / "hey-kalki-1.wav", dtype="float32")
    assert len(det.push(x) + det.push(np.zeros(4000, np.float32))) == 1          # and it still works
