"""Hard-negative mining: find speech/noise the current model wrongly fires on and turn it into training data.

    python mine_negatives.py --audio corpus_dir [--model models/kalki_fp32.onnx] [--threshold 0.5] [--tag r1]
Slides the model over every wav/flac under --audio (audio that must NOT contain the wake word, such as
LibriSpeech, Common Voice, podcasts, TV, room noise). Each false trigger becomes
data/negative/mined_<tag>_<n>.wav, which train.py treats as a hard negative (and never as a held-out voice).
Prints the false alarms per hour of the source audio, so it doubles as an evaluation on real recordings.
Retrain after mining, then mine again with --tag r2 to catch what is left.
"""
import argparse
import json
from pathlib import Path

import numpy as np
import onnxruntime as ort
import soundfile as sf

HERE = Path(__file__).parent
SR = 16000
WIN = int(1.5 * SR)
HOP = SR // 12
EXTS = {".wav", ".flac", ".ogg"}


def read_audio(path):
    a, sr = sf.read(path, dtype="float32")
    a = a.mean(1) if a.ndim > 1 else a
    if sr != SR and len(a) > 1:
        from scipy.signal import resample_poly
        g = np.gcd(int(sr), SR)
        a = resample_poly(a, SR // g, int(sr) // g).astype(np.float32)
    return a


def window_probs(sess, audio):
    """Probability that the wake word just ended, for windows of WIN samples every HOP samples."""
    starts = np.arange(0, len(audio) - WIN + 1, HOP)
    p = np.empty(len(starts), np.float32)
    for k, s in enumerate(starts):  # the exported GRU has batch size 1, same as live.py
        p[k] = sess.run(None, {"audio": audio[s:s + WIN][None]})[0].reshape(-1)[0]
    return starts, 1 / (1 + np.exp(-p))


def find_hits(starts, p, thr, min_gap=SR):
    """Peak window of every run of windows above thr; peaks closer than min_gap samples are merged."""
    hits, k = [], 0
    while k < len(p):
        if p[k] > thr:
            j = k
            while j + 1 < len(p) and p[j + 1] > thr:
                j += 1
            m = k + int(np.argmax(p[k:j + 1]))
            end = int(starts[m]) + WIN
            if hits and end - hits[-1][0] < min_gap:
                if p[m] > hits[-1][1]:
                    hits[-1] = (end, float(p[m]))
            else:
                hits.append((end, float(p[m])))
            k = j + 1
        else:
            k += 1
    return hits


def clip_for(audio, end, length=int(1.2 * SR), tail=int(0.1 * SR)):
    """The audio leading up to a false trigger (what the model thought was the wake word)."""
    return audio[max(end - length, 0): min(end + tail, len(audio))]


def mine(model, audio_dir, out_dir, thr=0.5, tag="r1", max_clips=2000, per_file=3, max_hours=None, log=print):
    sess = ort.InferenceSession(str(model), providers=["CPUExecutionProvider"])
    out_dir = Path(out_dir); out_dir.mkdir(parents=True, exist_ok=True)
    files = sorted(p for p in Path(audio_dir).rglob("*") if p.suffix.lower() in EXTS)
    hours, saved, total_hits, found = 0.0, 0, 0, []
    for f in files:
        if saved >= max_clips or (max_hours and hours >= max_hours):
            break
        try:
            a = read_audio(f)
        except Exception as e:
            log(f"skip {f.name}: {e}")
            continue
        if len(a) < WIN:
            continue
        hours += len(a) / SR / 3600
        starts, p = window_probs(sess, a)
        hits = find_hits(starts, p, thr)
        total_hits += len(hits)
        for end, score in sorted(hits, key=lambda h: -h[1])[:per_file]:
            if saved >= max_clips:
                break
            name = f"mined_{tag}_{saved}.wav"
            sf.write(out_dir / name, clip_for(a, end), SR)
            found.append({"clip": name, "source": str(f), "end_s": end / SR, "score": score})
            saved += 1
    return {"hours": hours, "false_alarms": total_hits, "false_alarms_per_hour": total_hits / max(hours, 1e-9),
            "threshold": thr, "saved": saved, "clips": found}


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--audio", required=True, help="folder of audio that does NOT contain the wake word")
    ap.add_argument("--model", default=str(HERE / "models" / "kalki_fp32.onnx"))
    ap.add_argument("--out", default=str(HERE / "data" / "negative"))
    ap.add_argument("--threshold", type=float, default=0.5)
    ap.add_argument("--tag", default="r1", help="round name; becomes the clip's 'voice', so use no underscores")
    ap.add_argument("--max-clips", type=int, default=2000)
    ap.add_argument("--per-file", type=int, default=3, help="keep at most this many hits per source file")
    ap.add_argument("--max-hours", type=float, help="stop after this much audio")
    ap.add_argument("--report", help="write the numbers (and where each clip came from) here")
    a = ap.parse_args()
    r = mine(a.model, a.audio, a.out, a.threshold, a.tag, a.max_clips, a.per_file, a.max_hours)
    print(f"{r['hours']:.2f} h scanned, {r['false_alarms']} false alarms at threshold {a.threshold} "
          f"({r['false_alarms_per_hour']:.1f}/h), saved {r['saved']} clips")
    if a.report:
        Path(a.report).write_text(json.dumps(r, indent=2))
