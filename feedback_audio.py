"""Score real recordings that people attach to a feedback issue (nothing is kept afterwards).

    python feedback_audio.py score --body issue.md [--model models/kalki_int8.onnx] [--out comment.md]
    python feedback_audio.py install --zip clips.zip [--data data]     # consented clips -> training data

`score` finds the .wav/.flac/.ogg/.zip attachments in the issue text, downloads them (GitHub attachment hosts only, size
capped), decides which are wake words and which are everyday audio, runs the model over them exactly as live.py would
and writes a markdown comment. Labels come from a clip's folder or file name (positive/, pos_*, negative/, neg_*), else from
the "What is in the audio" answer on the form.
`install` unpacks a zip with that same layout into data/positive/pos_real_N.wav and data/negative/neg_real_N.wav, which
train.py trains on 20x (positives) and 5x (negatives) more often than synthetic clips.
"""
import argparse
import io
import re
import tempfile
import urllib.request
import zipfile
from pathlib import Path, PurePosixPath
from urllib.parse import urlparse

import numpy as np
import soundfile as sf

import score_real

HERE = Path(__file__).parent
AUDIO_EXT = {".wav", ".flac", ".ogg"}
ALLOWED_HOSTS = {"github.com", "user-images.githubusercontent.com", "objects.githubusercontent.com"}
MAX_FILE = 25 * 1024 * 1024       # per download
MAX_UNZIPPED = 120 * 1024 * 1024  # all clips from one zip
MAX_MEMBERS = 200
MAX_SECONDS = 600                 # per clip, so a long recording cannot stall the job
URL_RE = re.compile(r"https://[^\s)\]>\"']+")


def sections(body):
    """Issue-form answers as {label: text} (GitHub renders each field as '### Label' then the answer)."""
    parts = re.split(r"(?m)^### +(.+?)\s*$", body)
    return {parts[i].strip(): parts[i + 1].strip() for i in range(1, len(parts) - 1, 2)}


def attachment_urls(body):
    """Links to audio or zip files hosted by GitHub, in order, without repeats; other links are ignored."""
    out = []
    for u in URL_RE.findall(body):
        u = u.rstrip(".,;")
        p = urlparse(u)
        if p.netloc not in ALLOWED_HOSTS or (p.netloc == "github.com" and not p.path.startswith("/user-attachments/")):
            continue
        # extensionless /user-attachments/assets/<uuid> links are sniffed after download
        ok = p.path.startswith("/user-attachments/assets/") or Path(p.path).suffix.lower() in AUDIO_EXT | {".zip"}
        if ok and u not in out:
            out.append(u)
    return out


def kind_from_name(name):
    """'positive' or 'negative' from a path inside a zip or a file name, else None."""
    parts = [x.lower() for x in PurePosixPath(name.replace("\\", "/")).parts]
    for part in reversed(parts):
        stem = PurePosixPath(part).stem
        if stem in ("positive", "positives", "pos") or stem.startswith(("pos_", "pos-", "positive_", "positive-")):
            return "positive"
        if stem in ("negative", "negatives", "neg") or stem.startswith(("neg_", "neg-", "negative_", "negative-")):
            return "negative"
    return None


def kind_from_form(answers):
    """The 'What is in the audio' dropdown: 'Me saying ...' -> positive, 'Everyday audio ...' -> negative, 'A mix' -> None."""
    v = next((t for k, t in answers.items() if "what is in the audio" in k.lower()), "").lower()
    return "positive" if v.startswith("me saying") else "negative" if v.startswith("everyday") else None


def read_clip(data, name):
    """Decode audio bytes to 16 kHz mono float32, or None if it is not usable audio."""
    try:
        with sf.SoundFile(io.BytesIO(data)) as f:
            if f.frames / max(f.samplerate, 1) > MAX_SECONDS:
                return None
            x = f.read(dtype="float32")
            sr = f.samplerate
    except Exception:
        return None
    x = x.mean(1) if x.ndim > 1 else x
    if sr != score_real.SR:
        from math import gcd
        from scipy.signal import resample_poly
        g = gcd(sr, score_real.SR)
        x = resample_poly(x, score_real.SR // g, sr // g).astype(np.float32)
    return x if len(x) >= score_real.SR // 5 else None


def clips_from_zip(data, default_kind=None):
    """[(name, kind, samples)] from zip bytes. Member names are only labels: nothing is extracted to disk by name."""
    out, total = [], 0
    with zipfile.ZipFile(io.BytesIO(data)) as z:
        infos = [i for i in z.infolist() if not i.is_dir() and Path(i.filename).suffix.lower() in AUDIO_EXT]
        for i in infos[:MAX_MEMBERS]:
            total += i.file_size
            if total > MAX_UNZIPPED or i.file_size > MAX_FILE:
                break
            x = read_clip(z.read(i), i.filename)
            if x is not None:
                out.append((i.filename, kind_from_name(i.filename) or default_kind, x))
    return out


def download(url):
    req = urllib.request.Request(url, headers={"User-Agent": "kalki-feedback-scorer"})
    with urllib.request.urlopen(req, timeout=60) as r:
        final = urlparse(r.geturl())
        if final.scheme != "https" or not (final.netloc == "github.com" or final.netloc.endswith(".githubusercontent.com")):
            raise ValueError("redirected away from GitHub")
        data = r.read(MAX_FILE + 1)
    if len(data) > MAX_FILE:
        raise ValueError("file too large")
    return data


def load_attachments(body, fetch=download):
    """Download and decode everything attached to the issue. Returns (clips, notes)."""
    default_kind = kind_from_form(sections(body))
    clips, notes = [], []
    for u in attachment_urls(body):
        name = PurePosixPath(urlparse(u).path).name or "attachment"
        try:
            data = fetch(u)
        except Exception as e:
            notes.append(f"could not download `{name}` ({e})")
            continue
        if zipfile.is_zipfile(io.BytesIO(data)):
            try:
                found = clips_from_zip(data, default_kind)
            except Exception as e:
                notes.append(f"could not read zip `{name}` ({e})")
                continue
            clips += found
            if not found:
                notes.append(f"`{name}` has no usable audio")
        else:
            x = read_clip(data, name)
            if x is None:
                notes.append(f"`{name}` is not usable audio (or longer than {MAX_SECONDS // 60} min)")
            else:
                clips.append((name, kind_from_name(name) or default_kind, x))
    return clips, notes


def score_clips(clips, model, thresholds=(0.5, 0.7, 0.8, 0.9)):
    """Score decoded clips with score_real's rows. Returns (rows, peaks, counts) or None without both kinds."""
    with tempfile.TemporaryDirectory() as tmp:
        d = {"positive": [], "negative": []}
        for i, (name, kind, x) in enumerate(clips):
            if kind in d:
                p = Path(tmp) / kind / f"{i:03d}_{Path(name).stem[:40]}.wav"
                p.parent.mkdir(exist_ok=True)
                sf.write(p, x, score_real.SR)
                d[kind].append(p)
        if not d["positive"] and not d["negative"]:
            return None
        rows, peaks = score_real.score_model(model, d["positive"], d["negative"], list(thresholds))
    return rows, peaks, {k: len(v) for k, v in d.items()}


def report(clips, notes, rows_peaks, answers=None):
    skipped = [n for n, k, _ in clips if k is None]
    lines = ["### Scoring your recordings", ""]
    if rows_peaks is None:
        lines.append("I could not find labelled audio. Attach `.wav`/`.flac`/`.ogg` files or a `.zip`, and say what is in them "
                     "(or name files `pos_*.wav` / `neg_*.wav`, or use `positive/` and `negative/` folders in the zip).")
    else:
        rows, _, n = rows_peaks
        lines += [f"{n['positive']} wake-word clips and {n['negative']} everyday-audio clips, scored with `models/kalki_int8.onnx` "
                  "the way `live.py` would hear them.", "",
                  "| Threshold | Wake words caught | 95% interval | False alarms | False alarms / hour |",
                  "|----------:|------------------:|-------------:|-------------:|--------------------:|"]
        for r in rows:
            lo, hi = score_real.wilson(r["caught"], r["positives"]) if r["positives"] else (0, 0)
            caught = f"{r['caught']}/{r['positives']}" if r["positives"] else "n/a"
            fa = f"{r['false_alarms']} in {r['negative_hours'] * 60:.1f} min" if n["negative"] else "n/a"
            fph = f"{r['false_alarms_per_hour']:.0f}" if n["negative"] else "n/a"
            lines.append(f"| {r['threshold']} | {caught} | " + (f"{lo:.0%} to {hi:.0%}" if r["positives"] else "n/a") + f" | {fa} | {fph} |")
        main = next((r for r in rows if r["threshold"] == 0.8), rows[0])
        if main["missed_files"]:
            lines += ["", "Missed at 0.8: " + ", ".join(f"`{m.split('_', 1)[-1]}`" for m in main["missed_files"][:10])]
        if main["false_alarm_files"]:
            lines += ["", "False alarms at 0.8: " + ", ".join(f"`{k.split('_', 1)[-1]}` x{v}" for k, v in list(main["false_alarm_files"].items())[:10])]
        lines += ["", "With few clips each number is rough (see the interval); use it to compare against other models, not as a grade."]
    if skipped:
        lines += ["", f"{len(skipped)} clip(s) had no label and were skipped: " + ", ".join(f"`{s}`" for s in skipped[:5])]
    if notes:
        lines += ["", "Notes: " + "; ".join(notes)]
    lines += ["", "_Audio is deleted after scoring; only these numbers are kept._"]
    return "\n".join(lines)


def cmd_score(a):
    body = Path(a.body).read_text()
    clips, notes = load_attachments(body)
    res = score_clips(clips, a.model) if any(k for _, k, _ in clips) else None
    md = report(clips, notes, res)
    if a.out:
        Path(a.out).write_text(md)
    print(md)


def cmd_install(a):
    data = Path(a.zip).read_bytes()
    clips = [c for c in clips_from_zip(data) if c[1]]
    data_dir = Path(a.data)
    n = {"positive": 0, "negative": 0}
    for _, kind, x in clips:
        folder = data_dir / kind
        folder.mkdir(parents=True, exist_ok=True)
        tag = "pos" if kind == "positive" else "neg"
        i = score_real_next_index(folder, tag)
        sf.write(folder / f"{tag}_real_{i}.wav", x, score_real.SR)
        n[kind] += 1
    print(f"installed {n['positive']} positive and {n['negative']} negative clips into {data_dir}")


def score_real_next_index(folder, tag):
    used = [int(p.stem.rsplit("_", 1)[1]) for p in folder.glob(f"{tag}_real_*.wav")]
    return max(used, default=-1) + 1


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest="cmd", required=True)
    s = sub.add_parser("score")
    s.add_argument("--body", required=True, help="file holding the issue text")
    s.add_argument("--model", default=str(HERE / "models" / "kalki_int8.onnx"))
    s.add_argument("--out")
    s.set_defaults(fn=cmd_score)
    i = sub.add_parser("install")
    i.add_argument("--zip", required=True)
    i.add_argument("--data", default=str(HERE / "data"))
    i.set_defaults(fn=cmd_install)
    a = ap.parse_args()
    a.fn(a)
