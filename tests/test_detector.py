"""Regression test: the shipped model must fire on the sample "hey kalki" clips and stay quiet on the rest.

These are the same 7 clips the browser demo plays (synthetic voices held out of training).
Run with:  pytest
"""
from pathlib import Path

import numpy as np
import pytest
import soundfile as sf

from detector import SR, WakeWordDetector

ROOT = Path(__file__).resolve().parent.parent
CLIPS = ROOT / "docs" / "clips"
SHOULD_TRIGGER = ["hey-kalki-1", "hey-kalki-2", "hey-kalki-3"]
SHOULD_STAY_QUIET = ["hey-kali", "hey-google", "kalki-alone", "sentence"]
MODELS = ["kalki_int4.onnx", "kalki_int8.onnx", "kalki_fp16.onnx", "kalki_fp32.onnx"]


def run(name, chunk=1280, model="kalki_int8.onnx", threshold=0.8):
    x, sr = sf.read(CLIPS / f"{name}.wav", dtype="float32")
    assert sr == SR
    det = WakeWordDetector(ROOT / "models" / model, threshold=threshold)
    hits = []
    for i in range(0, len(x), chunk):
        hits += det.push(x[i:i + chunk])
    hits += det.push(np.zeros(4000, np.float32))  # let the last window finish
    return hits


@pytest.mark.parametrize("model", MODELS)
@pytest.mark.parametrize("name", SHOULD_TRIGGER)
def test_fires_on_hey_kalki(name, model):
    assert len(run(name, model=model)) == 1


@pytest.mark.parametrize("model", MODELS)
@pytest.mark.parametrize("name", SHOULD_STAY_QUIET)
def test_quiet_on_everything_else(name, model):
    assert run(name, model=model) == []


@pytest.mark.parametrize("chunk", [1, 333, 1280, 5000, 24000])
def test_result_does_not_depend_on_chunk_size(chunk):
    assert len(run("hey-kalki-1", chunk=chunk)) == 1
    assert run("hey-google", chunk=chunk) == []


def test_silence_never_fires():
    det = WakeWordDetector()
    assert det.push(np.zeros(SR * 10, np.float32)) == []


def test_cooldown_prevents_double_fire():
    x, _ = sf.read(CLIPS / "hey-kalki-1.wav", dtype="float32")
    det = WakeWordDetector()
    # same word twice within the 1.5 s cooldown window counts once
    hits = det.push(x) + det.push(np.zeros(2000, np.float32)) + det.push(x)
    assert len(hits) >= 1
    assert all(b.time - a.time > det.cooldown for a, b in zip(hits, hits[1:]))


def test_high_threshold_is_stricter():
    # at an impossible threshold nothing fires
    assert run("hey-kalki-1", threshold=1.01) == []
