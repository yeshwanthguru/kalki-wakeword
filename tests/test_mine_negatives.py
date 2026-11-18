from pathlib import Path

import numpy as np
import soundfile as sf

import mine_negatives as mn
from augment import background, channel, impulsive, load_noise_bank

SR = mn.SR
repo_root = Path(__file__).resolve().parent.parent


def test_find_hits_merges_runs_and_close_peaks():
    starts = np.arange(10) * mn.HOP
    p = np.array([0, .9, .95, 0, 0, .8, 0, 0, 0, 0], np.float32)
    hits = mn.find_hits(starts, p, 0.5, min_gap=SR)
    assert len(hits) == 1 and abs(hits[0][1] - .95) < 1e-6
    assert len(mn.find_hits(starts, p, 0.5, min_gap=1)) == 2
    assert mn.find_hits(starts, p, 0.99) == []


def test_clip_for_is_bounded():
    a = np.zeros(5 * SR, np.float32)
    assert len(mn.clip_for(a, 2 * SR)) == int(1.3 * SR)
    assert len(mn.clip_for(a, SR // 2)) == SR // 2 + int(0.1 * SR)


def test_mine_saves_trainable_names(tmp_path):
    clips = repo_root / "docs" / "clips"
    audio = tmp_path / "corpus"; audio.mkdir()
    for n in ("hey-kali", "hey-google", "sentence"):
        a, sr = sf.read(clips / f"{n}.wav", dtype="float32")
        sf.write(audio / f"{n}.wav", np.concatenate([np.zeros(SR, np.float32), a, np.zeros(SR, np.float32)]), sr)
    out = tmp_path / "neg"
    r = mn.mine(repo_root / "models" / "kalki_fp32.onnx", audio, out, thr=0.0, tag="t1")  # thr 0: everything fires
    names = sorted(p.name for p in out.glob("*.wav"))
    assert r["saved"] == len(names) > 0 and r["hours"] > 0
    assert all(n.startswith("mined_t1_") and n.split("_")[1] == "t1" for n in names)  # train.voice_of() == "t1"


def test_mine_quiet_audio_finds_nothing(tmp_path):
    audio = tmp_path / "c"; audio.mkdir()
    sf.write(audio / "silence.wav", np.zeros(4 * SR, np.float32), SR)
    r = mn.mine(repo_root / "models" / "kalki_fp32.onnx", audio, tmp_path / "o", thr=0.5)
    assert r["saved"] == 0 and r["false_alarms_per_hour"] == 0


def test_augment_backgrounds_are_finite_and_bounded():
    rng = np.random.default_rng(0)
    colored = lambda n, r: r.standard_normal(n).astype(np.float32)
    speech = [rng.standard_normal(SR * 2).astype(np.float32) for _ in range(4)]
    bank = [rng.standard_normal(SR).astype(np.float32)]
    for _ in range(60):
        x = background(int(1.5 * SR), rng, colored, bank, speech)
        assert x.shape == (int(1.5 * SR),) and np.isfinite(x).all() and np.abs(x).max() <= 1.0001
    assert np.abs(impulsive(SR, rng)).max() <= 1.0001
    assert channel(speech[0], rng, p=1.0).shape == speech[0].shape


def test_noise_bank_loads_and_resamples(tmp_path):
    sf.write(tmp_path / "a.wav", np.random.randn(8000 * 2).astype(np.float32) * 0.1, 8000)
    sf.write(tmp_path / "short.wav", np.zeros(100, np.float32), 16000)
    bank = load_noise_bank(tmp_path)
    assert len(bank) == 1 and abs(len(bank[0]) - 32000) < 5
