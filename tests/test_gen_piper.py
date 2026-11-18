import sys
import types
from pathlib import Path

import numpy as np
import pytest
import soundfile as sf

import gen_piper as gp


def test_voice_tags_have_no_underscore_and_start_with_pp():
    t = gp.voice_tag("en_GB-vctk-medium", 12)
    assert t == "ppenGBvctkmediums12" and "_" not in t and gp.voice_tag("en_US-amy-medium") == "ppenUSamymedium"


def test_pick_speakers_is_reproducible_and_bounded():
    a = gp.pick_speakers(109, 30)
    assert a == gp.pick_speakers(109, 30) and len(set(a)) == 30 and max(a) < 109
    assert gp.pick_speakers(5, 30) == [0, 1, 2, 3, 4] and gp.pick_speakers(1, 30) == [None]


def test_plan_matches_gen_tts_mix():
    jobs = gp.plan("m", [None, 3], ["hey kalki"], ["hey kali"], ["go home"], per_voice=12)
    by = lambda s, t, spk: [j for j in jobs if j[0] == s and j[1] == t and j[3] == spk]
    assert len(by("positive", "pos", 3)) == 12 and len(by("negative", "hard", 3)) == 6 and len(by("negative", "sent", None)) == 6
    assert {j[4] for j in jobs} == {"ppm", "ppms3"}


class FakeVoice:
    class config:
        sample_rate = 22050
        num_speakers = 3

    def synthesize(self, text, cfg):
        t = np.arange(int(0.6 * 22050)) / 22050
        yield types.SimpleNamespace(audio_float_array=(0.3 * np.sin(2 * np.pi * 220 * t)).astype(np.float32))


def test_run_writes_16k_clips_with_held_out_safe_names(tmp_path, monkeypatch):
    fake = types.ModuleType("piper")
    fake.SynthesisConfig = lambda **k: types.SimpleNamespace(**k)
    fake.PiperVoice = types.SimpleNamespace(load=lambda p: FakeVoice())
    monkeypatch.setitem(sys.modules, "piper", fake)
    models = tmp_path / "models"; models.mkdir()
    (models / "en_GB-test-medium.onnx").write_bytes(b"x")
    n = gp.run(models, ["en_GB-test-medium", "missing-voice"], speakers=2, per_voice=4, out=tmp_path / "data", log=lambda *_: None)
    files = sorted((tmp_path / "data").rglob("*.wav"))
    assert n == len(files) == 2 * (4 + 2 + 2)
    x, sr = sf.read(files[0])
    assert sr == 16000 and len(x) > 16000 // 5
    assert all(p.stem.split("_")[1].startswith("pp") for p in files)   # voice_of() -> never a held-out voice
    assert gp.run(models, ["en_GB-test-medium"], 2, 4, out=tmp_path / "data", log=lambda *_: None) == 0  # rerun is a no-op
