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
