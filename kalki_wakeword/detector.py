"""Streaming "hey kalki" detector you can drop into any Python app.

    from kalki_wakeword import WakeWordDetector
    det = WakeWordDetector()                  # loads the bundled kalki_int8.onnx
    for hit in det.push(samples):             # float32 mono 16 kHz, any chunk size
        print(f"heard it at {hit.time:.2f}s (p={hit.prob:.2f})")
"""
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import onnxruntime as ort

SR = 16000
WIN = 24000   # model input: 1.5 s
HOP = 1280    # scored every 80 ms
DEFAULT_MODEL = Path(__file__).parent / "models" / "kalki_int8.onnx"   # kept identical to models/kalki_int8.onnx (tests check it)


@dataclass
class Detection:
    time: float   # seconds of audio fed so far when it fired
    prob: float


class WakeWordDetector:
    def __init__(self, model_path=DEFAULT_MODEL, threshold=0.8, hits=2, cooldown=1.5):
        so = ort.SessionOptions()
        so.intra_op_num_threads = 1
        self.sess = ort.InferenceSession(str(model_path), so, providers=["CPUExecutionProvider"])
        self.threshold, self.hits, self.cooldown = threshold, hits, cooldown
        self.reset()

    def reset(self):
        self.buf = np.zeros(WIN, np.float32)
        self.pending = np.zeros(0, np.float32)
        self.streak, self.clock, self.last_fire, self.last_prob = 0, 0.0, -1e9, 0.0

    def prob(self, window):
        logit = self.sess.run(None, {"audio": window[None].astype(np.float32)})[0][0]
        return float(1 / (1 + np.exp(-logit)))

    def push(self, samples):
        """Feed 16 kHz mono float32 audio in [-1, 1]. Returns the detections that fired in this chunk."""
        self.pending = np.concatenate([self.pending, np.asarray(samples, np.float32).ravel()])
        fired = []
        while len(self.pending) >= HOP:
            chunk, self.pending = self.pending[:HOP], self.pending[HOP:]
            self.buf = np.concatenate([self.buf[HOP:], chunk])
            self.last_prob = self.prob(self.buf)
            self.clock += HOP / SR
            self.streak = self.streak + 1 if self.last_prob > self.threshold else 0
            if self.streak >= self.hits and self.clock - self.last_fire > self.cooldown:
                self.last_fire = self.clock
                fired.append(Detection(self.clock, self.last_prob))
        return fired
