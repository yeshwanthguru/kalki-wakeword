"""Benchmark the 'hey kalki' detector against alternative models trained on the same data.

    python benchmark.py train --variant mlp|cnn|gru_large   # trains a comparison model
    python benchmark.py eval                                # scores everything, writes docs/
All models see the same synthetic data, the same held-out voices and the same fixed test set.
"""
import argparse
import json
import random
import time
from pathlib import Path

import numpy as np
import onnxruntime as ort
import torch
import torch.nn as nn
import torch.nn.functional as F

from model import HOP, N_MELS, SR, WIN, LogMel, WakeNet
from train import Sampler, load, place, voice_of

HERE = Path(__file__).parent
OUT, DOCS = HERE / "out", HERE / "docs"
DEV = "cuda" if torch.cuda.is_available() else "cpu"


class MLPNet(nn.Module):
    """Naive baseline: average + spread of each mel band over time, then an MLP (ignores word order)."""
    def __init__(self):
        super().__init__()
        self.mel = LogMel()
        self.norm = nn.BatchNorm1d(N_MELS)
        self.mlp = nn.Sequential(nn.Linear(2 * N_MELS, 128), nn.ReLU(), nn.Linear(128, 1))

    def forward(self, wav):
        x = self.norm(self.mel(wav))
        return self.mlp(torch.cat([x.mean(-1), x.std(-1)], -1)).squeeze(-1)


class CNNNet(nn.Module):
    """Same conv stack as WakeNet but pooled over time instead of a GRU."""
    def __init__(self, ch=48):
        super().__init__()
        self.mel = LogMel()
        self.norm = nn.BatchNorm1d(N_MELS)
        def block(i, o): return nn.Sequential(nn.Conv1d(i, o, 5, padding=2), nn.BatchNorm1d(o), nn.ReLU())
        self.conv = nn.Sequential(block(N_MELS, ch), nn.MaxPool1d(2), block(ch, ch), nn.MaxPool1d(2), block(ch, ch))
        self.head = nn.Linear(2 * ch, 1)

    def forward(self, wav):
        x = self.conv(self.norm(self.mel(wav)))
        return self.head(torch.cat([x.mean(-1), x.amax(-1)], -1)).squeeze(-1)


VARIANTS = {
    "mlp": ("MLP baseline", MLPNet),
    "cnn": ("CNN, no GRU", CNNNet),
    "kalki": ("CNN+GRU (ours)", WakeNet),
    "gru_large": ("CNN+GRU, 4x wider", lambda: WakeNet(ch=96, hidden=128)),
}


def splits():
    pos, neg = load("positive"), load("negative")
    voices = sorted({voice_of(p) for p in pos} - {"real"})
    held = set(random.Random(0).sample(voices, max(len(voices) // 8, 1)))
    sp = lambda d, h: {p: v for p, v in d.items() if (voice_of(p) in held) == h}
    return sp(pos, False), sp(neg, False), sp(pos, True), sp(neg, True)


def train(variant, steps=6000, batch=128, lr=2e-3):
    tp, tn, _, _ = splits()
    s = Sampler(tp, tn, 1)
    model = VARIANTS[variant][1]().to(DEV)
    opt = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-2)
    sched = torch.optim.lr_scheduler.OneCycleLR(opt, lr, total_steps=steps)
    for step in range(1, steps + 1):
        x, y = s.batch(batch, DEV)
        loss = F.binary_cross_entropy_with_logits(model(x), y)
        opt.zero_grad(); loss.backward(); opt.step(); sched.step()
        if step % 500 == 0:
            print(variant, step, round(loss.item(), 4), flush=True)
    torch.save(model.state_dict(), OUT / f"bench_{variant}.pt")


def get_model(variant):
    m = VARIANTS[variant][1]()
    ck = OUT / ("kalki.pt" if variant == "kalki" else f"bench_{variant}.pt")
    m.load_state_dict(torch.load(ck, map_location="cpu"))
    return m.eval()


@torch.no_grad()
def scores(model, x, bs=500):
    model = model.to(DEV)
    return np.concatenate([torch.sigmoid(model(x[i:i + bs].to(DEV))).cpu().numpy() for i in range(0, len(x), bs)])


def roc_metrics(p_pos, p_neg):
    allp = np.concatenate([p_pos, p_neg])
    ranks = allp.argsort().argsort() + 1
    auc = (ranks[:len(p_pos)].sum() - len(p_pos) * (len(p_pos) + 1) / 2) / (len(p_pos) * len(p_neg))
    thr1 = np.quantile(p_neg, 0.99)
    ths = np.unique(np.concatenate([[0, 1], allp]))
    far = np.array([(p_neg >= t).mean() for t in ths]); frr = np.array([(p_pos < t).mean() for t in ths])
    eer = far[np.argmin(np.abs(far - frr))]
    return {"auc": float(auc), "recall@0.5": float((p_pos > 0.5).mean()), "fpr@0.5": float((p_neg > 0.5).mean()),
            "frr@1%far": float((p_pos < thr1).mean()), "eer": float(eer)}, (far, frr)


def latency_ms(model, variant):
    path = OUT / f"bench_{variant}.onnx"
    torch.onnx.export(model.cpu(), torch.zeros(1, WIN), path, input_names=["audio"], output_names=["logit"], opset_version=17)
    so = ort.SessionOptions(); so.intra_op_num_threads = 1
    sess = ort.InferenceSession(str(path), so)
    x = np.random.randn(1, WIN).astype(np.float32) * 0.1
    for _ in range(20): sess.run(None, {"audio": x})
    t = time.perf_counter()
    for _ in range(200): sess.run(None, {"audio": x})
    ms = (time.perf_counter() - t) / 200 * 1000
    size = path.stat().st_size + (Path(str(path) + ".data").stat().st_size if Path(str(path) + ".data").exists() else 0)
    return ms, size / 1024


def build_stream(vpos, vneg, minutes=15, seed=7):
    """Long audio of held-out clips with gaps, background noise, and known wake word times."""
    from train import noise
    rng = np.random.default_rng(seed)
    pos, neg = list(vpos.values()), list(vneg.values())
    n = int(minutes * 60 * SR)
    stream = noise(n, rng) * 10 ** (-42 / 20)
    t, ends = int(SR), []
    while t < n - 3 * SR:
        is_pos = rng.random() < 0.12
        clip = (pos if is_pos else neg)[rng.integers(len(pos if is_pos else neg))]
        clip = clip / (np.abs(clip).max() + 1e-6) * 10 ** (rng.uniform(-24, -6) / 20)
        stream[t:t + len(clip)] += clip[: n - t]
        t += len(clip)
        if is_pos:
            ends.append(t)
        t += int(rng.uniform(0.3, 2.0) * SR)
    return np.clip(stream, -1, 1).astype(np.float32), ends


def trigger_stats(p, starts, n_samples, ends, thr=0.8, hits=2):
    """Turn per-window probabilities into (fraction of wake words caught, false alarms per hour)."""
    trig, streak, last = [], 0, -1e9
    for k, pk in enumerate(p):
        streak = streak + 1 if pk > thr else 0
        tend = starts[k] + WIN
        if streak >= hits and tend - last > 1.5 * SR:
            trig.append(tend); last = tend
    found, fa = set(), 0
    for tt in trig:
        m = [i for i, e in enumerate(ends) if e - 0.3 * SR <= tt <= e + 0.8 * SR]
        if m: found.add(m[0])
        else: fa += 1
    return len(found) / max(len(ends), 1), fa / (n_samples / SR / 3600)


@torch.no_grad()
def stream_eval(model, stream, ends, thr=0.8, hits=2, hop=SR // 12):
    model = model.to(DEV)
    s = torch.from_numpy(stream)
    starts = np.arange(0, len(stream) - WIN, hop)
    p = np.concatenate([torch.sigmoid(model(torch.stack([s[i:i + WIN] for i in starts[j:j + 256]]).to(DEV))).cpu().numpy()
                        for j in range(0, len(starts), 256)])
    return trigger_stats(p, starts, len(stream), ends, thr, hits)


def evaluate():
    import matplotlib; matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    _, _, vp, vn = splits()
    ts = Sampler(vp, vn, 123)
    x, y = ts.batch(4000, "cpu")
    y = y.numpy() > 0.5
    stream, ends = build_stream(vp, vn)
    print(f"test set: {y.sum()} pos / {(~y).sum()} neg windows; stream: {len(stream)/SR/60:.0f} min, {len(ends)} wake words")
    DOCS.mkdir(exist_ok=True)
    results, curves = {}, {}
    for v, (name, _) in VARIANTS.items():
        m = get_model(v)
        p = scores(m, x)
        r, curves[v] = roc_metrics(p[y], p[~y])
        r["params_k"] = sum(q.numel() for q in m.parameters()) / 1e3
        r["latency_ms"], r["size_kb"] = latency_ms(m, v)
        r["stream_detect@0.8"], r["stream_fa_per_hour@0.8"] = stream_eval(m, stream, ends)
        r["name"] = name
        results[v] = r
        print(v, {k: round(a, 4) if isinstance(a, float) else a for k, a in r.items()})
    (DOCS / "benchmark.json").write_text(json.dumps(results, indent=2))

    fig, ax = plt.subplots(1, 2, figsize=(11, 4.2))
    for v, (far, frr) in curves.items():
        ax[0].plot(np.maximum(far, 1e-4), 1 - frr, label=f"{results[v]['name']} (AUC {results[v]['auc']:.3f})",
                   lw=2.5 if v == "kalki" else 1.5)
    ax[0].set_xscale("log"); ax[0].set_xlabel("false-accept rate"); ax[0].set_ylabel("recall")
    ax[0].set_title("ROC on held-out voices"); ax[0].legend(fontsize=8); ax[0].grid(alpha=.3)
    names = [results[v]["name"] for v in results]
    ax[1].bar(names, [results[v]["latency_ms"] for v in results], color=["#999", "#999", "#2a7", "#999"])
    ax[1].set_ylabel("ms per inference (1 CPU thread)"); ax[1].set_title("Speed")
    plt.setp(ax[1].get_xticklabels(), rotation=20, ha="right", fontsize=8)
    plt.tight_layout(); plt.savefig(DOCS / "benchmark.png", dpi=130)


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("cmd", choices=["train", "eval"])
    ap.add_argument("--variant", choices=["mlp", "cnn", "gru_large"])
    a = ap.parse_args()
    train(a.variant) if a.cmd == "train" else evaluate()
