"""Streaming "hey kalki" detector you can drop into any Python app.

    from kalki_wakeword import WakeWordDetector
    det = WakeWordDetector()                  # loads the bundled 36 KB model (falls back to the 232 KB one)
    for hit in det.push(samples):             # float32 mono 16 kHz, any chunk size
        print(f"heard it at {hit.time:.2f}s (p={hit.prob:.2f})")
"""
import warnings
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import onnxruntime as ort

SR = 16000
WIN = 24000   # model input: 1.5 s
HOP = 1280    # scored every 80 ms
INT4_MODEL = Path(__file__).parent / "models" / "kalki_int4.onnx"   # 36 KB, needs onnxruntime >= 1.20; the default
DEFAULT_MODEL = Path(__file__).parent / "models" / "kalki_int8.onnx"   # 232 KB, loads everywhere; the fallback
# Both are kept identical to models/kalki_int4.onnx and models/kalki_int8.onnx (tests check it).


# Named trade-offs between missed wake words and false alarms: (probability threshold, consecutive windows that must exceed it).
# Measured for the shipped model on eval-set-v2 (229 wake words, 60 min of held-out synthetic speech; see tune_trigger.py):
#   sensitive  (0.6, 2): about 98% caught, 44 false alarms/hour     balanced (0.8, 2): about 97% caught, 27/hour (the default)
#   strict     (0.9, 3): about 95% caught, 13 false alarms/hour
# Real microphones and rooms differ, so treat these as a starting point and confirm with score_real.py on your own recordings.
SENSITIVITY = {"sensitive": (0.6, 2), "balanced": (0.8, 2), "strict": (0.9, 3)}


@dataclass
class Detection:
    time: float   # seconds of audio fed so far when it fired
    prob: float


class WakeWordDetector:
    def __init__(self, model_path=None, threshold=None, hits=None, cooldown=1.5, sensitivity="balanced"):
        """model_path: a .onnx file. Default: the bundled 36 KB int4 model, or the 232 KB int8 model if the installed onnxruntime
        cannot load it (it needs 1.20 or newer). The file actually used is in `self.model_path`.
        sensitivity: 'sensitive', 'balanced' (default) or 'strict' picks the threshold and hits; passing threshold or hits
        yourself overrides that part of the preset."""
        if sensitivity not in SENSITIVITY:
            raise ValueError(f"sensitivity must be one of {sorted(SENSITIVITY)}, got {sensitivity!r}")
        preset_threshold, preset_hits = SENSITIVITY[sensitivity]
        threshold = preset_threshold if threshold is None else threshold
        hits = preset_hits if hits is None else hits
        self.sess, self.model_path = self._open(model_path)
        self.threshold, self.hits, self.cooldown = threshold, hits, cooldown
        self.reset()

    @staticmethod
    def _open(model_path):
        def session(path):
            so = ort.SessionOptions()
            so.intra_op_num_threads = 1
            return ort.InferenceSession(str(path), so, providers=["CPUExecutionProvider"])
        if model_path is not None:
            return session(model_path), Path(model_path)
        try:
            return session(INT4_MODEL), INT4_MODEL
        except Exception as e:   # an onnxruntime too old for 4-bit weights
            warnings.warn(f"could not load the 36 KB int4 model ({str(e).splitlines()[0][:120]}); using the 232 KB int8 model. "
                          "Upgrade onnxruntime to 1.20 or newer to use the smaller one.", RuntimeWarning, stacklevel=3)
            return session(DEFAULT_MODEL), DEFAULT_MODEL

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
