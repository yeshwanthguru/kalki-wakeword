"""Wake word -> speech-to-text -> your LLM, as a starting point. Say "hey kalki", then speak your request.

    pip install faster-whisper
    python examples/with_llm.py

The wake word part is tested. The speech-to-text and LLM glue below is an untested sketch:
replace ask_llm() with a call to your own model (a local function, or an HTTP endpoint).
"""
import sys
import time
from pathlib import Path

import numpy as np
import sounddevice as sd

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from detector import HOP, SR, WakeWordDetector  # noqa: E402


def ask_llm(prompt: str) -> str:
    """Plug your model in here. Example for any OpenAI-compatible server:

        import json, urllib.request
        req = urllib.request.Request("http://localhost:8000/v1/chat/completions",
            json.dumps({"model": "mine", "messages": [{"role": "user", "content": prompt}]}).encode(),
            {"Content-Type": "application/json"})
        return json.load(urllib.request.urlopen(req))["choices"][0]["message"]["content"]
    """
    return f"(placeholder reply) You said: {prompt}"


def record_utterance(max_s=8.0, silence_s=0.9, floor=0.01):
    """Record until the speaker stops (simple loudness-based endpointing)."""
    chunks, quiet, t0 = [], 0.0, time.time()
    with sd.InputStream(samplerate=SR, channels=1, blocksize=HOP, dtype="float32") as s:
        while time.time() - t0 < max_s:
            x, _ = s.read(HOP)
            chunks.append(x[:, 0])
            quiet = quiet + HOP / SR if np.abs(x).mean() < floor else 0.0
            if quiet >= silence_s and time.time() - t0 > 1.0:
                break
    return np.concatenate(chunks)


def main():
    from faster_whisper import WhisperModel
    stt = WhisperModel("base.en", compute_type="int8")
    det = WakeWordDetector()
    print("say 'hey kalki' ...")
    with sd.InputStream(samplerate=SR, channels=1, blocksize=HOP, dtype="float32") as s:
        while True:
            x, _ = s.read(HOP)
            if det.push(x[:, 0]):
                print("listening for your request ...")
            else:
                continue
            s.stop()
            audio = record_utterance()
            segments, _ = stt.transcribe(audio, language="en")
            text = " ".join(seg.text for seg in segments).strip()
            print("you:", text)
            if text:
                print("llm:", ask_llm(text))
            det.reset()
            s.start()
            print("say 'hey kalki' ...")


if __name__ == "__main__":
    main()
