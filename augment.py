"""Background-noise augmentation for training (numpy only).

Adds to the synthetic colored noise in train.noise(): noise recorded from the real world (any wavs in
data/noise/), babble made of other speech, impulsive sounds, and microphone / channel coloring.
"""
from pathlib import Path

import numpy as np
import soundfile as sf
from scipy.signal import butter, sosfilt

SR = 16000


def load_noise_bank(folder, max_files=2000):
    """Real noise recordings (any sample rate/channels) as 16 kHz mono float32 arrays; [] if none."""
    bank = []
    for p in sorted(Path(folder).glob("**/*.wav"))[:max_files]:
        try:
            a, sr = sf.read(p, dtype="float32")
        except Exception:
            continue
        a = a.mean(1) if a.ndim > 1 else a
        if sr != SR and len(a) > 1:
            a = np.interp(np.arange(0, len(a), sr / SR), np.arange(len(a)), a).astype(np.float32)
        if len(a) >= SR // 2:
            bank.append(a)
    return bank


def crop(a, n, rng):
    """Random n-sample piece of a (looped if it is too short)."""
    if len(a) < n:
        a = np.tile(a, n // len(a) + 1)
    s = int(rng.integers(0, len(a) - n + 1))
    return a[s:s + n]


def normalize(x):
    return x / (np.abs(x).max() + 1e-6)


def impulsive(n, rng):
    """A few clicks / door knocks / claps: short decaying noise bursts."""
    x = np.zeros(n, np.float32)
    for _ in range(int(rng.integers(1, 6))):
        L = int(rng.uniform(0.005, 0.12) * SR)
        burst = rng.standard_normal(L) * np.exp(-np.linspace(0, rng.uniform(4, 9), L))
        s = int(rng.integers(0, max(n - L, 1)))
        x[s:s + L] += (burst * rng.uniform(0.3, 1.0))[: n - s]
    return normalize(x)


def babble(n, rng, speech):
    """3-6 overlapping speech clips -> unintelligible crowd noise."""
    x = np.zeros(n, np.float32)
    for _ in range(int(rng.integers(3, 7))):
        x += crop(normalize(speech[int(rng.integers(len(speech)))]), n, rng) * rng.uniform(0.3, 1.0)
    return normalize(x)


def channel(x, rng, p=0.5):
    """Mimic a cheap mic or phone line: band-limit, or random tilt, or crude bit depth."""
    if rng.random() > p:
        return x
    kind = rng.choice(["lowpass", "highpass", "bandpass", "bitcrush"])
    if kind == "bitcrush":
        levels = 2 ** int(rng.integers(6, 11))
        return (np.round(x * levels) / levels).astype(np.float32)
    if kind == "lowpass":
        sos = butter(2, rng.uniform(2500, 6500), "lowpass", fs=SR, output="sos")
    elif kind == "highpass":
        sos = butter(2, rng.uniform(150, 600), "highpass", fs=SR, output="sos")
    else:
        sos = butter(2, [rng.uniform(250, 500), rng.uniform(2800, 3800)], "bandpass", fs=SR, output="sos")
    y = sosfilt(sos, x).astype(np.float32)
    return y * (np.abs(x).max() / (np.abs(y).max() + 1e-6))


def background(n, rng, colored, bank=(), speech=()):
    """One background of length n, peak-normalized. `colored` is train.noise (synthetic noise kinds)."""
    choices = ["colored", "impulsive"] + (["real"] * 3 if len(bank) else []) + (["babble"] if len(speech) else [])
    kind = rng.choice(choices)
    if kind == "real":
        x = normalize(crop(bank[int(rng.integers(len(bank)))], n, rng))
    elif kind == "impulsive":
        x = impulsive(n, rng) + 0.05 * colored(n, rng)
    elif kind == "babble":
        x = babble(n, rng, speech)
    else:
        x = colored(n, rng)
    return channel(normalize(x), rng)
