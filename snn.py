"""Spiking ("neuromorphic") version of the wake word network.

    python snn.py train --steps 6000      # trains on data/ (CPU), writes out/kalki_snn.pt
    python snn.py eval                    # scores it on the same held-out voices / frozen eval set as ci_eval.py
                                          #   and counts operations; writes snn_report.json / snn_report.md
    python snn.py export                  # out/kalki_snn.onnx (time loop unrolled, plain ops only)

Same log-mel front end as WakeNet. After it, the network runs one time step per (pooled) frame:

    frame (40 mel values, analog) -> dense layer -> LIF neurons -> spikes (0/1)
        -> dense + recurrent layer -> LIF neurons -> spikes -> leaky integrator readout -> logit

LIF = leaky integrate-and-fire: v = beta * v + input; spike when v >= 1; subtract 1 after a spike. Everything after
the first layer only moves 0/1 spikes, so a synapse costs one addition and only when its neuron spikes. That is
where an event-driven chip (or a C loop that skips silent neurons) saves work. Training uses a surrogate gradient
(hard threshold forward, smooth sigmoid slope backward).

Nothing here runs on neuromorphic hardware: the report counts operations, it does not measure power.
"""
import argparse
import json
import sys
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from model import N_MELS, WIN, LogMel, spec_augment

HERE = Path(__file__).parent
OUT = HERE / "out"
POOL = 2          # frames averaged per time step: 148 frames of 10 ms -> 74 steps of 20 ms
SLOPE = 5.0       # sharpness of the surrogate gradient


def spike(v, training):
    """1.0 where v >= 0 (v is membrane minus threshold). Backward pass uses a sigmoid slope instead of the step."""
    hard = (v >= 0).to(v.dtype)
    if not training:
        return hard
    soft = torch.sigmoid(SLOPE * v)
    return hard + soft - soft.detach()


class SpikingWakeNet(nn.Module):
    def __init__(self, h1=192, h2=128):
        super().__init__()
        self.mel = LogMel()
        self.norm = nn.BatchNorm1d(N_MELS)
        self.fc1 = nn.Linear(N_MELS, h1)
        self.fc2 = nn.Linear(h1, h2, bias=False)
        self.rec = nn.Linear(h2, h2, bias=False)
        self.out = nn.Linear(h2, 1, bias=False)
        self.bias = nn.Parameter(torch.zeros(1))
        # per-neuron leak, kept in (0, 1) through a sigmoid; start with memories of a few steps up to ~20
        self.raw_beta1 = nn.Parameter(torch.empty(h1).uniform_(-0.5, 2.5))
        self.raw_beta2 = nn.Parameter(torch.empty(h2).uniform_(-0.5, 2.5))
        self.raw_beta_out = nn.Parameter(torch.tensor(2.0))
        self.h1, self.h2 = h1, h2
        self.spec_augment = False
        self.record = False       # when True, forward() stores spike counts in self.stats (used by op_counts)
        self.stats = {}

    def forward(self, wav):
        x = self.norm(self.mel(wav))
        if self.training and self.spec_augment:
            x = spec_augment(x)
        x = F.avg_pool1d(x, POOL).transpose(1, 2)          # (B, steps, 40)
        B, T, _ = x.shape
        b1, b2, bo = torch.sigmoid(self.raw_beta1), torch.sigmoid(self.raw_beta2), torch.sigmoid(self.raw_beta_out)
        v1 = x.new_zeros(B, self.h1)
        v2 = x.new_zeros(B, self.h2)
        s2 = x.new_zeros(B, self.h2)
        y = x.new_zeros(B, 1)
        cur1 = self.fc1(x)                                  # the first layer sees analog input, all steps at once
        n1 = n2 = 0.0
        rate = 0.0
        for t in range(T):
            v1 = b1 * v1 + cur1[:, t]
            s1 = spike(v1 - 1.0, self.training)
            v1 = v1 - s1
            v2 = b2 * v2 + self.fc2(s1) + self.rec(s2)
            s2 = spike(v2 - 1.0, self.training)
            v2 = v2 - s2
            y = bo * y + self.out(s2)
            if self.training or self.record:
                n1 = n1 + s1.sum(); n2 = n2 + s2.sum()
                rate = rate + s1.mean() + s2.mean()
        if self.training or self.record:
            self.stats = {"spikes1": n1, "spikes2": n2, "rate": rate / (2 * T), "steps": T, "batch": B}
        return (y + self.bias).squeeze(-1)


def op_counts(model, x, bs=256):
    """Average per 1.5 s window on audio batch x: spike rates and the work after the front end.

    dense_macs: multiply-adds that happen every step regardless (first layer, leak updates).
    synaptic_adds: additions that happen only when a presynaptic neuron spikes (fan-out of each spike).
    """
    model.eval(); model.record = True
    s1 = s2 = 0.0; steps = None
    with torch.no_grad():
        for i in range(0, len(x), bs):
            xb = x[i:i + bs]
            model(xb)
            s1 += float(model.stats["spikes1"]); s2 += float(model.stats["spikes2"]); steps = model.stats["steps"]
    model.record = False
    n = len(x)
    s1, s2 = s1 / n, s2 / n                                  # spikes per window
    h1, h2 = model.h1, model.h2
    dense = steps * (N_MELS * h1 + h1 + h2 + 1)                 # fc1 on analog frames + one leak multiply per neuron
    adds = s1 * h2 + s2 * h2 + s2 * 1                           # fc2 per layer-1 spike, rec + out per layer-2 spike
    return {"steps": steps, "spikes_per_window": {"layer1": s1, "layer2": s2},
            "rate_per_neuron_per_step": {"layer1": s1 / (h1 * steps), "layer2": s2 / (h2 * steps)},
            "dense_macs": dense, "synaptic_adds": adds,
            "dense_synapses_if_not_event_driven": steps * (h1 * h2 + h2 * h2 + h2)}


def export_onnx(model, path):
    model.eval().cpu()
    torch.onnx.export(model, torch.zeros(1, WIN), str(path), input_names=["audio"], output_names=["logit"],
                      opset_version=17)   # batch 1, like the GRU export (that is how a detector runs it)


# ---- training and evaluation (need data/, see gen_tts.py) -------------------------------------------------------

def cmd_train(a):
    import copy
    from train import DEVICE, Sampler, batches, default_workers, load, load_noise_bank, seed_everything, split_data
    from train import HERE as TRAIN_HERE
    if a.seed is not None:
        seed_everything(a.seed)
    seed = 1 if a.seed is None else a.seed
    pos, neg = load("positive"), load("negative")
    assert pos, "no clips: run gen_tts.py first"
    tp, tn, vp, vn = split_data(pos, neg)
    bank = load_noise_bank(str(TRAIN_HERE / "data" / "noise"))
    train_s, val_s = Sampler(tp, tn, seed, augment=True, noise_bank=bank), Sampler(vp, vn, 2)
    model = SpikingWakeNet().to(DEVICE)
    model.spec_augment = a.specaugment
    print(f"{sum(p.numel() for p in model.parameters()) / 1e3:.1f}K parameters")
    opt = torch.optim.AdamW(model.parameters(), lr=a.lr, weight_decay=1e-2)
    sched = torch.optim.lr_scheduler.OneCycleLR(opt, a.lr, total_steps=a.steps)
    workers = default_workers() if a.workers is None else a.workers
    if workers > 0:
        import os
        torch.set_num_threads(max((os.cpu_count() or 1) - workers, 1))
    stream = batches(train_s, a.batch, workers, seed=seed)
    for step in range(1, a.steps + 1):
        x, y = next(stream)
        logit = model(x)
        loss = F.binary_cross_entropy_with_logits(logit, y)
        reg = a.rate_penalty * model.stats["rate"] if a.rate_penalty else 0.0
        opt.zero_grad(); (loss + reg).backward(); opt.step(); sched.step()
        if step % 250 == 0:
            model.eval()
            with torch.no_grad():
                xv, yv = val_s.batch(1024, DEVICE)
                pred = torch.sigmoid(model(xv)) > 0.5
            model.train()
            print(f"step {step} loss {loss.item():.4f} rate {float(model.stats['rate']):.3f} | held-out recall "
                  f"{pred[yv == 1].float().mean():.3f} false-pos {pred[yv == 0].float().mean():.3f}", flush=True)
    OUT.mkdir(exist_ok=True)
    torch.save(model.eval().state_dict(), OUT / "kalki_snn.pt")
    print("saved", OUT / "kalki_snn.pt")


def cmd_eval(a):
    import eval_set
    from benchmark import SR, build_stream, roc_metrics, scores, splits, stream_eval
    from train import Sampler
    model = SpikingWakeNet(); model.load_state_dict(torch.load(OUT / "kalki_snn.pt", map_location="cpu")); model.eval()
    _, _, vp, vn = splits()
    x, y = Sampler(vp, vn, 123).batch(4000, "cpu")
    y = y.numpy() > 0.5
    p = scores(model, x)
    r, _ = roc_metrics(p[y], p[~y])
    stream, ends = build_stream(vp, vn, minutes=eval_set.stream_minutes())
    r["stream"] = {}
    for thr in (0.5, 0.8, 0.9):
        det, fa = stream_eval(model, stream, ends, thr=thr)
        r["stream"][str(thr)] = {"detected": det, "false_alarms_per_hour": fa}
    r["ops"] = op_counts(model, x)
    r["params"] = sum(q.numel() for q in model.parameters())
    r["eval_set"] = eval_set.version() or "regenerated"
    base = json.loads((HERE / "docs" / "baseline.json").read_text()).get("int4")
    s = r["stream"]["0.8"]
    o = r["ops"]
    lines = ["| model | AUC | recall@0.5 | FPR@0.5 | caught@0.8 | false alarms/h @0.8 |", "|---|---:|---:|---:|---:|---:|",
             f"| spiking | {r['auc']:.4f} | {r['recall@0.5']:.3f} | {r['fpr@0.5']:.3f} | {s['detected']:.0%} | "
             f"{s['false_alarms_per_hour']:.0f} |"]
    if base:
        b = base["stream"]["0.8"]
        lines.append(f"| int4 GRU (committed) | {base['auc']:.4f} | {base['recall@0.5']:.3f} | {base['fpr@0.5']:.3f} | "
                     f"{b['detected']:.0%} | {b['false_alarms_per_hour']:.0f} |")
    lines += ["", f"Evaluation set: `{r['eval_set']}`. Parameters: {r['params']:,}.", "",
              "Work per 1.5 s window after the front end (the GRU network is about 3.5M multiply-adds):", "",
              f"- spikes per window: layer 1 {o['spikes_per_window']['layer1']:.0f}, layer 2 {o['spikes_per_window']['layer2']:.0f} "
              f"(firing rate {o['rate_per_neuron_per_step']['layer1']:.1%} / {o['rate_per_neuron_per_step']['layer2']:.1%} of neurons per step)",
              f"- dense multiply-adds: {o['dense_macs']:,.0f} (first layer on analog frames and leak updates)",
              f"- event-driven additions: {o['synaptic_adds']:,.0f} (instead of {o['dense_synapses_if_not_event_driven']:,} if every "
              "synapse were evaluated every step)", "",
              "These are operation counts, not measured power."]
    (HERE / "snn_report.json").write_text(json.dumps(r, indent=2))
    (HERE / "snn_report.md").write_text("\n".join(lines))
    print("\n".join(lines))


def cmd_export(a):
    m = SpikingWakeNet(); m.load_state_dict(torch.load(OUT / "kalki_snn.pt", map_location="cpu"))
    export_onnx(m, OUT / "kalki_snn.onnx")
    print("wrote", OUT / "kalki_snn.onnx")


def main(argv=None):
    ap = argparse.ArgumentParser()
    sub = ap.add_subparsers(dest="cmd", required=True)
    t = sub.add_parser("train")
    t.add_argument("--steps", type=int, default=6000)
    t.add_argument("--batch", type=int, default=128)
    t.add_argument("--lr", type=float, default=3e-3)
    t.add_argument("--seed", type=int, default=None)
    t.add_argument("--workers", type=int, default=None)
    t.add_argument("--specaugment", action="store_true")
    t.add_argument("--rate-penalty", type=float, default=0.0, help="weight on the mean firing rate (higher = sparser)")
    t.set_defaults(fn=cmd_train)
    sub.add_parser("eval").set_defaults(fn=cmd_eval)
    sub.add_parser("export").set_defaults(fn=cmd_export)
    a = ap.parse_args(argv)
    a.fn(a)


if __name__ == "__main__":
    main()
