"""score_real.py: the scoring helpers and the split command, using the demo clips as stand-ins for real recordings."""
import shutil
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import onnxruntime as ort
import soundfile as sf

import score_real

ROOT = Path(__file__).resolve().parent.parent
CLIPS = ROOT / "docs" / "clips"
MODEL = ROOT / "models" / "kalki_int8.onnx"


def session():
    return ort.InferenceSession(str(MODEL), providers=["CPUExecutionProvider"])


def test_wake_word_clip_fires_and_others_do_not():
    sess = session()
    rng = np.random.default_rng(0)
    hit = score_real.probabilities(sess, score_real.padded(score_real.load_wav(CLIPS / "hey-kalki-1.wav"), rng))
    miss = score_real.probabilities(sess, score_real.load_wav(CLIPS / "hey-google.wav"))
    assert score_real.triggers(hit, 0.8)
    assert score_real.triggers(miss, 0.8) == []


def test_wilson_interval_widens_with_few_samples():
    lo10, hi10 = score_real.wilson(10, 10)
    lo100, hi100 = score_real.wilson(100, 100)
    assert hi10 == 1.0 and lo10 < lo100 < 1.0
    assert score_real.wilson(0, 0) == (0.0, 0.0)


def test_split_moves_clips_but_leaves_some_for_training(tmp_path):
    for kind, tag in (("positive", "pos"), ("negative", "neg")):
        d = tmp_path / "data" / kind
        d.mkdir(parents=True)
        for i in range(6):
            shutil.copy(CLIPS / "hey-kalki-1.wav", d / f"{tag}_real_{i}.wav")
    args = SimpleNamespace(data=str(tmp_path / "data"), dir=str(tmp_path / "real_test"), n_pos=4, n_neg=100, seed=0)
    score_real.cmd_split(args)
    assert len(list((tmp_path / "real_test" / "positive").glob("*.wav"))) == 4
    assert len(list((tmp_path / "data" / "positive").glob("*.wav"))) == 2
    # asking for more than exist still keeps one clip for training
    assert len(list((tmp_path / "data" / "negative").glob("*.wav"))) == 1


def test_resamples_other_rates_to_16k(tmp_path):
    x = (np.sin(2 * np.pi * 440 * np.arange(48000) / 48000) * 0.3).astype(np.float32)
    sf.write(tmp_path / "tone.wav", x, 48000)
    assert abs(len(score_real.load_wav(tmp_path / "tone.wav")) - 16000) <= 1
