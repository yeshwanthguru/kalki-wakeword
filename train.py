"""Train the 'hey kalki' detector on synthetic clips.

    python wakeword/train.py --steps 6000
Outputs wakeword/out/kalki.pt and wakeword/out/kalki.onnx.
"""
import argparse
import random
from pathlib import Path

import numpy as np
import soundfile as sf
import torch
import torch.nn.functional as F
from scipy.signal import fftconvolve

from augment import background, load_noise_bank
from model import SR, WIN, WakeNet

HERE = Path(__file__).parent
OUT = HERE / "out"


def load(sub):
    return {p: sf.read(p, dtype="float32")[0] for p in sorted((HERE / "data" / sub).glob("*.wav"))}


def voice_of(p):
    return p.stem.split("_")[1]


def noise(n, rng):
    kind = rng.choice(["white", "pink", "brown", "hum", "quiet"])
    w = rng.standard_normal(n).astype(np.float32)
    if kind == "white":
        x = w
    elif kind == "quiet":
        x = w * 0.05
    else:
        spec = np.fft.rfft(w)
        f = np.maximum(np.arange(len(spec)), 1)
        x = np.fft.irfft(spec / (f ** (0.5 if kind == "pink" else 1.0)), n).astype(np.float32)
        if kind == "hum":
            t = np.arange(n) / SR
            x = x + 3 * np.sin(2 * np.pi * rng.choice([50, 100, 120]) * t) * x.std()
    return x / (np.abs(x).max() + 1e-6)


def stretch(a, rng, lo=0.85, hi=1.15):
    r = rng.uniform(lo, hi)
    return np.interp(np.arange(0, len(a), r), np.arange(len(a)), a).astype(np.float32)


def reverb(a, rng, p=0.3):
    """Smear the clip with a synthetic room impulse response (decaying noise) some of the time."""
    if rng.random() > p:
        return a
    rt = rng.uniform(0.1, 0.6)
    t = np.arange(int(rt * SR)) / SR
    rir = rng.standard_normal(len(t)) * np.exp(-6.9 * t / rt)
    rir[0] = 1.0
    wet = fftconvolve(a, rir.astype(np.float32))[: len(a)]
    return (wet * (np.abs(a).max() / (np.abs(wet).max() + 1e-6))).astype(np.float32)


def fx(a, rng):
    return reverb(stretch(a, rng), rng)


def place(canvas, clip, end):
    """Put clip so it ends at `end` (may be cut off on the left)."""
    start = end - len(clip)
    lo, hi = max(start, 0), min(end, len(canvas))
    if hi > lo:
        canvas[lo:hi] += clip[lo - start: hi - start]


class Sampler:
    def __init__(self, pos, neg, seed, augment=False, noise_bank=()):
        # Your own recordings (voice "real", see record_real.py) are rare, so show them more often.
        rep = lambda d, k: [a for p, a in d.items() for _ in range(k if voice_of(p) == "real" else 1)]
        self.pos, self.neg = rep(pos, 20), rep(neg, 5)
        # hand-written near-misses and clips mined from real audio by mine_negatives.py
        self.hard = [a for p, a in neg.items() if p.stem.startswith(("hard", "mined"))]
        self.augment, self.noise_bank = augment, noise_bank
        self.speech = [a for p, a in neg.items() if p.stem.startswith(("sent", "mined"))]
        self.rng = np.random.default_rng(seed)

    def one(self, positive):
        rng = self.rng
        x = np.zeros(WIN, np.float32)
        bg = (background(WIN, rng, noise, self.noise_bank, self.speech) if self.augment else noise(WIN, rng)) \
            * 10 ** (rng.uniform(-45, -20) / 20)
        # speech in the background
        if rng.random() < 0.4:
            for _ in range(rng.integers(1, 3)):
                place(bg, self.neg[rng.integers(len(self.neg))] * 10 ** (rng.uniform(-35, -15) / 20), int(rng.integers(0, WIN + SR)))
        if positive:
            clip = fx(self.pos[rng.integers(len(self.pos))], rng)
            clip = clip / (np.abs(clip).max() + 1e-6) * 10 ** (rng.uniform(-30, -3) / 20)
            place(x, clip, WIN - int(rng.uniform(0, 0.25) * SR))
        else:
            r = rng.random()
            if r < 0.15:
                pass                                            # background only
            elif r < 0.30:                                      # wake word cut off mid-way
                clip = fx(self.pos[rng.integers(len(self.pos))], rng)
                clip = clip[: int(len(clip) * rng.uniform(0.2, 0.6))]
                place(x, clip / (np.abs(clip).max() + 1e-6) * 10 ** (rng.uniform(-30, -3) / 20), WIN - int(rng.uniform(0, 0.25) * SR))
            elif r < 0.45:                                      # wake word ended long ago, only trailing silence
                clip = fx(self.pos[rng.integers(len(self.pos))], rng)
                place(x, clip / (np.abs(clip).max() + 1e-6) * 10 ** (rng.uniform(-30, -3) / 20), WIN - int(rng.uniform(0.9, 1.4) * SR))
            else:
                pool = self.hard if rng.random() < 0.5 else self.neg
                clip = fx(pool[rng.integers(len(pool))], rng)
                place(x, clip / (np.abs(clip).max() + 1e-6) * 10 ** (rng.uniform(-30, -3) / 20), int(rng.integers(SR // 2, WIN + SR // 2)))
        x = x + bg
        return np.clip(x * 10 ** (rng.uniform(-6, 6) / 20), -1, 1)

    def batch(self, n, device):
        y = (np.arange(n) < n // 2)
        x = np.stack([self.one(p) for p in y]).astype(np.float32)
        return torch.from_numpy(x).to(device), torch.from_numpy(y.astype(np.float32)).to(device)


@torch.no_grad()
def evaluate(model, sampler, device, n=1024, thr=0.5):
    model.eval()
    x, y = sampler.batch(n, device)
    p = torch.sigmoid(model(x))
    model.train()
    pred = p > thr
    return (pred[y == 1].float().mean().item(), pred[y == 0].float().mean().item())


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--steps", type=int, default=6000)
    ap.add_argument("--batch", type=int, default=128)
    ap.add_argument("--lr", type=float, default=2e-3)
    ap.add_argument("--no-augment", action="store_true", help="only the original colored-noise backgrounds")
    ap.add_argument("--noise-dir", default=str(HERE / "data" / "noise"), help="real noise wavs (optional)")
    a = ap.parse_args()

    pos, neg = load("positive"), load("negative")
    assert pos, "no clips: run gen_tts.py first"
    voices = sorted({voice_of(p) for p in pos} - {"real"})  # real recordings always train
    held = set(random.Random(0).sample(voices, max(len(voices) // 8, 1)))
    split = lambda d, h: {p: v for p, v in d.items() if (voice_of(p) in held) == h}
    bank = [] if a.no_augment else load_noise_bank(a.noise_dir)
    train_s = Sampler(split(pos, False), split(neg, False), 1, augment=not a.no_augment, noise_bank=bank)
    val_s = Sampler(split(pos, True), split(neg, True), 2)
    print(f"{len(pos)} pos / {len(neg)} neg clips, held-out voices: {sorted(held)}, real noise files: {len(bank)}")

    device = "cuda" if torch.cuda.is_available() else "cpu"
    model = WakeNet().to(device)
    print(sum(p.numel() for p in model.parameters()) / 1e3, "K params")
    opt = torch.optim.AdamW(model.parameters(), lr=a.lr, weight_decay=1e-2)
    sched = torch.optim.lr_scheduler.OneCycleLR(opt, a.lr, total_steps=a.steps)

    for step in range(1, a.steps + 1):
        x, y = train_s.batch(a.batch, device)
        loss = F.binary_cross_entropy_with_logits(model(x), y)
        opt.zero_grad(); loss.backward(); opt.step(); sched.step()
        if step % 250 == 0:
            rec, fpr = evaluate(model, val_s, device)
            print(f"step {step} loss {loss.item():.4f} | held-out recall {rec:.3f} false-pos {fpr:.3f}", flush=True)

    OUT.mkdir(exist_ok=True)
    model.eval().cpu()
    torch.save(model.state_dict(), OUT / "kalki.pt")
    torch.onnx.export(model, torch.zeros(1, WIN), OUT / "kalki.onnx", input_names=["audio"],
                      output_names=["logit"], dynamic_axes={"audio": {0: "batch"}}, opset_version=17)
    print("saved", OUT)


if __name__ == "__main__":
    main()
