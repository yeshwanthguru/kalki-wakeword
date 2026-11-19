"""Train the 'hey kalki' detector on synthetic clips.

    python wakeword/train.py --steps 6000
Outputs wakeword/out/kalki.pt and wakeword/out/kalki.onnx.
"""
import argparse
import copy
import random
import sys
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
DEVICE = "cpu"   # training is CPU-only on purpose: the network is tiny and the audio augmentation is CPU work anyway


def load(sub):
    return {p: sf.read(p, dtype="float32")[0] for p in sorted((HERE / "data" / sub).glob("*.wav"))}


def voice_of(p):
    return p.stem.split("_")[1]


def split_data(pos, neg):
    """(train_pos, train_neg, heldout_pos, heldout_neg). With a published eval set (eval_set.py) the held-out data is that
    frozen audio and its voices are excluded from training; otherwise 1 in 8 of the generated voices is held out."""
    import eval_set
    if eval_set.exists():
        ev = eval_set.voices()
        keep = lambda d: {p: v for p, v in d.items() if voice_of(p) not in ev}
        vp, vn = eval_set.load_set()
        return keep(pos), keep(neg), vp, vn
    held = heldout_voices(pos)
    sp = lambda d, h: {p: v for p, v in d.items() if (voice_of(p) in held) == h}
    return sp(pos, False), sp(neg, False), sp(pos, True), sp(neg, True)


def heldout_voices(pos, fraction=8):
    """Voices kept out of training (1 in `fraction`, default 8). Real recordings and second-TTS voices ('pp...') always train, so the
    held-out set, and therefore every benchmark number, stays the same when more training voices are added."""
    voices = sorted({voice_of(p) for p in pos} - {"real"})
    voices = [v for v in voices if not v.startswith("pp")]
    return set(random.Random(0).sample(voices, max(len(voices) // fraction, 1)))


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


class _Stream(torch.utils.data.IterableDataset):
    """Endless augmented batches built on the CPU. Each DataLoader worker gets its own random stream."""
    def __init__(self, sampler, batch, seed):
        self.sampler, self.batch, self.seed = sampler, batch, seed

    def __iter__(self):
        info = torch.utils.data.get_worker_info()
        self.sampler.rng = np.random.default_rng([self.seed, info.id if info else 0])
        while True:
            yield self.sampler.batch(self.batch, "cpu")


def _one_thread(_):
    torch.set_num_threads(1)   # the main process owns the cores' BLAS threads; workers only run numpy


def seed_everything(seed):
    """Seed python, numpy and torch (network init, dropout-free but order-sensitive batches). Near-reproducible, not bit-exact:
    BLAS threading can still change float sums slightly."""
    random.seed(seed)
    np.random.seed(seed % 2**32)
    torch.manual_seed(seed)


def default_workers():
    """Worker processes for batch building. Forked on Linux only (spawn would copy the clips per worker).
    The cores are split about evenly between these workers and the training threads (measured on 4 cores: 2 + 2 is 2.4x
    faster than 0 + 4, 3 + 1 only 1.8x)."""
    import os
    cores = os.cpu_count() or 1
    if not sys.platform.startswith("linux") or cores < 3:
        return 0
    return cores // 2


def batches(sampler, batch, workers, seed=1):
    """Yield (x, y) CPU tensors. With workers > 0 the next batches are prepared in parallel while the network trains."""
    if workers <= 0:
        while True:
            yield sampler.batch(batch, DEVICE)
    loader = torch.utils.data.DataLoader(_Stream(sampler, batch, seed), batch_size=None, num_workers=workers,
                                         prefetch_factor=4, worker_init_fn=_one_thread)
    yield from loader


@torch.no_grad()
def update_ema(ema, model, decay):
    for e, p in zip(ema.parameters(), model.parameters()):
        e.lerp_(p, 1 - decay)
    for e, b in zip(ema.buffers(), model.buffers()):   # BatchNorm statistics follow the live model
        e.copy_(b)


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
    ap.add_argument("--specaugment", action="store_true", help="mask random mel bands / frames while training")
    ap.add_argument("--ema", type=float, default=0.0, help="keep an exponential moving average of the weights (e.g. 0.999) and save that")
    ap.add_argument("--label-smoothing", type=float, default=0.0, help="soften the 0/1 targets (e.g. 0.05)")
    ap.add_argument("--workers", type=int, default=None,
                    help="processes that build augmented batches in parallel with training (default: automatic, 0 = in the training loop)")
    ap.add_argument("--seed", type=int, default=None,
                    help="seed the network init and the batch streams (default: unseeded, runs differ)")
    ap.add_argument("--no-augment", action="store_true", help="only the original colored-noise backgrounds")
    ap.add_argument("--noise-dir", default=str(HERE / "data" / "noise"), help="real noise wavs (optional)")
    a = ap.parse_args()
    if a.seed is not None:
        seed_everything(a.seed)
    stream_seed = 1 if a.seed is None else a.seed

    pos, neg = load("positive"), load("negative")
    assert pos, "no clips: run gen_tts.py first"
    tp, tn, vp, vn = split_data(pos, neg)
    held = sorted({voice_of(p) for p in vp})
    bank = [] if a.no_augment else load_noise_bank(a.noise_dir)
    train_s = Sampler(tp, tn, stream_seed, augment=not a.no_augment, noise_bank=bank)
    val_s = Sampler(vp, vn, 2)
    print(f"{len(pos)} pos / {len(neg)} neg clips, held-out voices: {held}, real noise files: {len(bank)}")

    device = DEVICE
    model = WakeNet().to(device)
    model.spec_augment = a.specaugment
    ema = copy.deepcopy(model) if a.ema > 0 else None
    print(sum(p.numel() for p in model.parameters()) / 1e3, "K params")
    opt = torch.optim.AdamW(model.parameters(), lr=a.lr, weight_decay=1e-2)
    sched = torch.optim.lr_scheduler.OneCycleLR(opt, a.lr, total_steps=a.steps)

    workers = default_workers() if a.workers is None else a.workers
    if workers > 0:
        import os
        torch.set_num_threads(max((os.cpu_count() or 1) - workers, 1))   # leave the workers their cores
    print(f"building batches with {workers} worker process(es), {torch.get_num_threads()} training thread(s) on {device}")
    stream = batches(train_s, a.batch, workers, seed=stream_seed)
    for step in range(1, a.steps + 1):
        x, y = next(stream)
        target = y * (1 - a.label_smoothing) + a.label_smoothing / 2
        loss = F.binary_cross_entropy_with_logits(model(x), target)
        opt.zero_grad(); loss.backward(); opt.step(); sched.step()
        if ema is not None:
            update_ema(ema, model, min(a.ema, (1 + step) / (10 + step)))   # warm-up so early steps are not dominated by init
        if step % 250 == 0:
            rec, fpr = evaluate(ema if ema is not None else model, val_s, device)
            print(f"step {step} loss {loss.item():.4f} | held-out recall {rec:.3f} false-pos {fpr:.3f}", flush=True)

    OUT.mkdir(exist_ok=True)
    if ema is not None:
        model = ema
    model.eval().cpu()
    torch.save(model.state_dict(), OUT / "kalki.pt")
    torch.onnx.export(model, torch.zeros(1, WIN), OUT / "kalki.onnx", input_names=["audio"],
                      output_names=["logit"], dynamic_axes={"audio": {0: "batch"}}, opset_version=17)
    print("saved", OUT)


if __name__ == "__main__":
    main()
