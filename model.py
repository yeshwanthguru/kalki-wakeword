"""Wake word network: raw 16 kHz audio window -> probability that the wake word just ended.

The log-mel frontend is built from fixed Conv1d filters (a DFT as convolution) so the whole thing,
audio in to probability out, exports to ONNX with no stft op.
"""
import math

import torch
import torch.nn as nn

SR = 16000
WIN = int(1.5 * SR)      # model input length in samples
N_FFT = 400              # 25 ms
HOP = 160                # 10 ms
N_MELS = 40


def mel_filterbank(n_fft=N_FFT, n_mels=N_MELS, sr=SR, fmin=60.0, fmax=7600.0):
    def hz2mel(f): return 2595 * math.log10(1 + f / 700)
    def mel2hz(m): return 700 * (10 ** (m / 2595) - 1)
    pts = torch.tensor([mel2hz(hz2mel(fmin) + i * (hz2mel(fmax) - hz2mel(fmin)) / (n_mels + 1))
                        for i in range(n_mels + 2)])
    freqs = torch.linspace(0, sr / 2, n_fft // 2 + 1)
    fb = torch.zeros(n_mels, n_fft // 2 + 1)
    for i in range(n_mels):
        lo, c, hi = pts[i], pts[i + 1], pts[i + 2]
        fb[i] = torch.clamp(torch.minimum((freqs - lo) / (c - lo), (hi - freqs) / (hi - c)), min=0)
    return fb


class LogMel(nn.Module):
    def __init__(self):
        super().__init__()
        n = torch.arange(N_FFT).float()
        win = torch.hann_window(N_FFT, periodic=True)
        k = torch.arange(N_FFT // 2 + 1).float()[:, None]
        ang = 2 * math.pi * k * n[None] / N_FFT
        self.register_buffer("cos", (torch.cos(ang) * win).unsqueeze(1))   # (F,1,N_FFT)
        self.register_buffer("sin", (-torch.sin(ang) * win).unsqueeze(1))
        self.register_buffer("fb", mel_filterbank())

    def forward(self, wav):                       # (B, T)
        x = wav.unsqueeze(1)
        re, im = nn.functional.conv1d(x, self.cos, stride=HOP), nn.functional.conv1d(x, self.sin, stride=HOP)
        power = re ** 2 + im ** 2                 # (B, F, frames)
        mel = torch.matmul(self.fb, power)        # (B, mels, frames)
        return torch.log(mel + 1e-6)


class WakeNet(nn.Module):
    def __init__(self, ch=48, hidden=64):
        super().__init__()
        self.mel = LogMel()
        self.norm = nn.BatchNorm1d(N_MELS)
        def block(i, o): return nn.Sequential(nn.Conv1d(i, o, 5, padding=2), nn.BatchNorm1d(o), nn.ReLU())
        self.conv = nn.Sequential(block(N_MELS, ch), nn.MaxPool1d(2), block(ch, ch), nn.MaxPool1d(2), block(ch, ch))
        self.gru = nn.GRU(ch, hidden, batch_first=True)
        self.head = nn.Linear(hidden, 1)

    def forward(self, wav):
        x = self.norm(self.mel(wav))
        x = self.conv(x).transpose(1, 2)          # (B, frames/4, ch)
        _, h = self.gru(x)
        return self.head(h[-1]).squeeze(-1)       # logits; apply sigmoid for probability
