"""Second TTS source: Piper (open, offline) multi-speaker voices, to add voice and accent variety to the training set.

    pip install piper-tts
    python -m piper.download_voices --download-dir piper_models en_GB-vctk-medium en_US-libritts_r-medium
    python gen_piper.py --models-dir piper_models --voices en_GB-vctk-medium en_US-libritts_r-medium --speakers 30
Writes 16 kHz mono wavs next to gen_tts.py's, as data/{positive,negative}/<tag>_pp<model><speaker>_<n>.wav. The 'pp'
voice prefix keeps them out of the held-out set (train.heldout_voices), so benchmark numbers stay comparable with
edge-tts-only runs. Check each voice's licence (its MODEL_CARD) before using the result commercially: VCTK and LibriTTS-R
are CC BY 4.0, but some Piper voices (for example L2-ARCTIC) are non-commercial.
"""
import argparse
import random
import re
from math import gcd
from pathlib import Path

import numpy as np
import soundfile as sf
from scipy.signal import resample_poly

from gen_tts import HARD_NEGATIVE, POSITIVE, SENTENCES, SR, phrase_variants, trim

OUT = Path(__file__).parent / "data"


def voice_tag(model_name, speaker=None):
    """File-name 'voice' for a model/speaker. Must not contain '_' (train.voice_of splits on it)."""
    base = "pp" + re.sub(r"[^A-Za-z0-9]", "", model_name)
    return base if speaker is None else f"{base}s{speaker}"


def pick_speakers(num_speakers, n, seed=0):
    """Up to n distinct speaker ids, spread at random but reproducibly (None for single-speaker models)."""
    if num_speakers <= 1:
        return [None]
    return sorted(random.Random(seed).sample(range(num_speakers), min(n, num_speakers)))


def synth(voice, text, speaker, rng):
    """One utterance as 16 kHz mono float32, with random speaking rate and variability."""
    from piper import SynthesisConfig
    cfg = SynthesisConfig(speaker_id=speaker, length_scale=rng.uniform(0.8, 1.25),
                          noise_scale=rng.uniform(0.5, 0.9), noise_w_scale=rng.uniform(0.5, 1.0))
    chunks = [c.audio_float_array for c in voice.synthesize(text, cfg)]
    a = np.concatenate(chunks).astype(np.float32) if chunks else np.zeros(0, np.float32)
    sr = voice.config.sample_rate
    if sr != SR and len(a):
        g = gcd(sr, SR)
        a = resample_poly(a, SR // g, sr // g).astype(np.float32)
    return trim(a)


def plan(model_name, speakers, positives, hard, sentences, per_voice, seed=0):
    """[(subfolder, tag, text, speaker, voice_name, index)] for one model: same mix as gen_tts.py."""
    rng = random.Random(f"{seed}-{model_name}")
    jobs = []
    for spk in speakers:
        v = voice_tag(model_name, spk)
        for sub, tag, texts, n in (("positive", "pos", positives, per_voice),
                                   ("negative", "hard", hard, max(per_voice // 2, 1)),
                                   ("negative", "sent", sentences, max(per_voice // 2, 1))):
            jobs += [(sub, tag, rng.choice(texts), spk, v, i) for i in range(n)]
    return jobs


def run(models_dir, voices, speakers, per_voice, phrase=None, out=OUT, log=print):
    from piper import PiperVoice
    positives, hard = phrase_variants(phrase) if phrase else (POSITIVE, HARD_NEGATIVE)
    rng = random.Random(0)
    made = 0
    for name in voices:
        path = Path(models_dir) / f"{name}.onnx"
        if not path.exists():
            log(f"skip {name}: {path} not found")
            continue
        voice = PiperVoice.load(path)
        for sub, tag, text, spk, v, i in plan(name, pick_speakers(voice.config.num_speakers, speakers), positives, hard,
                                              SENTENCES, per_voice):
            dest = Path(out) / sub / f"{tag}_{v}_{i}.wav"
            if dest.exists():
                continue
            try:
                a = synth(voice, text, spk, rng)
            except Exception as e:
                log(f"fail: {name} speaker {spk}: {e}")
                continue
            if len(a) < SR // 5:
                continue
            dest.parent.mkdir(parents=True, exist_ok=True)
            sf.write(dest, a, SR)
            made += 1
        log(f"{name}: done ({made} clips so far)")
    return made


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--models-dir", default="piper_models")
    ap.add_argument("--voices", nargs="+", required=True, help="Piper model names, e.g. en_GB-vctk-medium")
    ap.add_argument("--speakers", type=int, default=30, help="speakers to use per multi-speaker model")
    ap.add_argument("--per-voice", type=int, default=12)
    ap.add_argument("--phrase", help="custom wake phrase (same meaning as gen_tts.py --phrase)")
    a = ap.parse_args()
    print("clips written:", run(a.models_dir, a.voices, a.speakers, a.per_voice, a.phrase))
