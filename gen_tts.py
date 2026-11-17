"""Synthesize training clips with Microsoft Edge TTS (needs internet, no API key).

Writes 16 kHz mono wavs to wakeword/data/{positive,negative}/.
    python wakeword/gen_tts.py --per-voice 12
"""
import argparse
import asyncio
import io
import random
from pathlib import Path

import edge_tts
import numpy as np
import soundfile as sf
from scipy.signal import resample_poly

OUT = Path(__file__).parent / "data"
SR = 16000

POSITIVE = ["hey kalki", "Hey Kalki", "hey, kalki", "Hey Kalki!", "hey kalki.", "hay kalki", "hey kalkee"]

# Near-misses and other speech the model must stay quiet for.
HARD_NEGATIVE = [
    "kalki", "hey", "hey kali", "hey calci", "hey Kelly", "hey Google", "hey Siri", "hey Alexa",
    "hey Jarvis", "ok Google", "hello kalki", "hi kalki", "hey kaki", "hey talk", "hey carlie",
    "hey call me", "okay kalki", "kalki avatar", "hey calcium", "a key", "hey Kelvin",
    "hey buddy", "hey there", "hey you", "hey Kiki", "hey Cali", "hey Karthik", "hey Kalyan",
]

SENTENCES = [
    "what time is it", "turn on the lights", "play some music", "the weather is nice today",
    "I need to buy groceries", "can you call me later", "how are you doing", "let's go for a walk",
    "the meeting starts at three", "I will be there soon", "please close the door",
    "what is the capital of France", "set an alarm for seven", "did you see the game last night",
    "I think it is going to rain", "open the window", "thank you very much", "see you tomorrow",
    "where did you put my keys", "this is a really long sentence about nothing in particular",
    "the quick brown fox jumps over the lazy dog", "I am learning to train a language model",
    "remind me to drink water", "my laptop is running very hot", "dinner is almost ready",
    "turn the volume down", "who is at the door", "it was a long day at work",
    "can you hear me", "one two three four five", "good morning everyone", "good night",
    "I forgot my charger at home", "the train is late again", "let me think about it",
    "send a message to my brother", "cooking rice takes about twenty minutes",
    "we should visit the temple this weekend", "the stock market closed higher today",
    "play the next song", "stop", "yes", "no", "maybe later", "I do not know", "wait a second",
]


async def synth(text, voice, rate, pitch):
    comm = edge_tts.Communicate(text, voice, rate=rate, pitch=pitch)
    buf = bytearray()
    async for chunk in comm.stream():
        if chunk["type"] == "audio":
            buf += chunk["data"]
    audio, sr = sf.read(io.BytesIO(bytes(buf)), dtype="float32")
    if audio.ndim > 1:
        audio = audio.mean(1)
    if sr != SR:
        g = np.gcd(sr, SR)
        audio = resample_poly(audio, SR // g, sr // g).astype(np.float32)
    return audio


def trim(a, thr=0.01, pad=int(0.05 * SR)):
    idx = np.where(np.abs(a) > thr * max(np.abs(a).max(), 1e-6))[0]
    if len(idx) == 0:
        return a
    return a[max(idx[0] - pad, 0): idx[-1] + pad]


async def worker(queue, sem, stats):
    while True:
        job = await queue.get()
        if job is None:
            return
        path, text, voice, rate, pitch = job
        try:
            async with sem:
                a = trim(await synth(text, voice, rate, pitch))
            sf.write(path, a, SR)
            stats["ok"] += 1
        except Exception as e:  # network hiccups, unsupported prosody on a voice
            stats["fail"] += 1
            if stats["fail"] <= 5:
                print("fail:", voice, text, e)


async def main(args):
    voices = [v for v in await edge_tts.list_voices() if v["Locale"].startswith("en-")]
    # Indian English first-class: the name is Indian, so is likely accent of the speaker.
    voices = [v["ShortName"] for v in voices]
    print(f"{len(voices)} English voices")
    random.seed(0)
    for sub in ("positive", "negative"):
        (OUT / sub).mkdir(parents=True, exist_ok=True)

    jobs = []

    def add(sub, tag, texts, n_per_voice):
        for v in voices:
            for i in range(n_per_voice):
                text = random.choice(texts)
                rate = f"{random.randint(-25, 25):+d}%"
                pitch = f"{random.randint(-30, 30):+d}Hz"
                p = OUT / sub / f"{tag}_{v}_{i}.wav"
                if not p.exists():
                    jobs.append((p, text, v, rate, pitch))

    add("positive", "pos", POSITIVE, args.per_voice)
    add("negative", "hard", HARD_NEGATIVE, max(args.per_voice // 2, 1))
    add("negative", "sent", SENTENCES, max(args.per_voice // 2, 1))
    random.shuffle(jobs)
    print(len(jobs), "clips to synthesize")

    queue, sem, stats = asyncio.Queue(), asyncio.Semaphore(args.concurrency), {"ok": 0, "fail": 0}
    for j in jobs:
        queue.put_nowait(j)
    for _ in range(args.concurrency):
        queue.put_nowait(None)
    await asyncio.gather(*[worker(queue, sem, stats) for _ in range(args.concurrency)])
    print(stats)


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--per-voice", type=int, default=12)
    ap.add_argument("--concurrency", type=int, default=8)
    asyncio.run(main(ap.parse_args()))
