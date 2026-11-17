"""Make the sample clips used by the browser demo (docs/clips). Voices are ones held out of training."""
import asyncio
from pathlib import Path

import numpy as np
import soundfile as sf

from gen_tts import SR, synth, trim

OUT = Path(__file__).parent / "docs" / "clips"
CLIPS = [  # (file, text, voice)
    ("hey-kalki-1", "hey kalki", "en-CA-ClaraNeural"),
    ("hey-kalki-2", "hey kalki", "en-KE-AsiliaNeural"),
    ("hey-kalki-3", "Hey Kalki!", "en-TZ-ElimuNeural"),
    ("hey-kali", "hey kali", "en-SG-LunaNeural"),
    ("hey-google", "hey Google", "en-CA-ClaraNeural"),
    ("kalki-alone", "kalki", "en-TZ-ElimuNeural"),
    ("sentence", "what time is it in London right now", "en-KE-AsiliaNeural"),
]


async def main():
    OUT.mkdir(parents=True, exist_ok=True)
    rng = np.random.default_rng(0)
    for name, text, voice in CLIPS:
        a = trim(await synth(text, voice, "+0%", "+0Hz"))
        a = a / np.abs(a).max() * 0.5
        quiet = lambda s: (rng.standard_normal(int(s * SR)) * 0.002).astype(np.float32)
        y = np.concatenate([quiet(0.7), a, quiet(1.3)])
        sf.write(OUT / f"{name}.wav", y, SR, subtype="PCM_16")
        print(name, round(len(y) / SR, 2), "s")


if __name__ == "__main__":
    asyncio.run(main())
