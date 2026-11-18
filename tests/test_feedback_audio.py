import io
import zipfile
from pathlib import Path

import numpy as np
import soundfile as sf

import feedback_audio as fa

ROOT = Path(__file__).resolve().parent.parent
CLIPS = ROOT / "docs" / "clips"

BODY = """### How did you test it?

Browser demo, with my microphone

### What is in the audio

Me saying "hey kalki" (wake word clips)

### Audio

[a.zip](https://github.com/user-attachments/files/123/a.zip) and
![x](https://github.com/user-attachments/assets/aaaa-bbbb) and https://evil.example.com/x.wav
and https://github.com/someone/else/raw/main/y.wav and https://github.com/user-attachments/files/1/clip.wav.
"""


def wav_bytes(x, sr=16000):
    b = io.BytesIO()
    sf.write(b, x, sr, format="WAV")
    return b.getvalue()


def test_sections_and_urls_only_github_attachments():
    s = fa.sections(BODY)
    assert s["How did you test it?"] == "Browser demo, with my microphone"
    assert fa.kind_from_form(s) == "positive"
    assert fa.attachment_urls(BODY) == [
        "https://github.com/user-attachments/files/123/a.zip",
        "https://github.com/user-attachments/assets/aaaa-bbbb",
        "https://github.com/user-attachments/files/1/clip.wav",
    ]


def test_kind_from_name_and_form():
    assert fa.kind_from_name("positive/a.wav") == "positive"
    assert fa.kind_from_name("take/neg_3.wav") == "negative"
    assert fa.kind_from_name("../../etc/passwd.wav") is None
    assert fa.kind_from_form({"What is in the audio": "Everyday audio without the wake word"}) == "negative"
    assert fa.kind_from_form({"What is in the audio": "A mix (name files pos_ and neg_)"}) is None


def test_zip_member_names_never_touch_the_filesystem(tmp_path):
    x = np.random.default_rng(0).standard_normal(16000).astype(np.float32) * 0.1
    z = io.BytesIO()
    with zipfile.ZipFile(z, "w") as zf:
        zf.writestr("../../evil/pos_1.wav", wav_bytes(x))
        zf.writestr("negative/b.wav", wav_bytes(x, 8000))
        zf.writestr("notes.txt", b"hi")
        zf.writestr("broken.wav", b"not audio")
    clips = fa.clips_from_zip(z.getvalue())
    assert sorted((Path(n).name, k) for n, k, _ in clips) == [("b.wav", "negative"), ("pos_1.wav", "positive")]
    assert not (tmp_path / "evil").exists()
    assert sorted(len(c[2]) for c in clips) == [16000, 32000]  # the 16000-sample 8 kHz clip is resampled to 16 kHz


def test_end_to_end_with_demo_clips(tmp_path):
    files = {"https://github.com/user-attachments/files/1/pos_a.wav": (CLIPS / "hey-kalki-1.wav").read_bytes(),
             "https://github.com/user-attachments/files/2/neg_b.wav": (CLIPS / "hey-google.wav").read_bytes(),
             "https://github.com/user-attachments/files/3/c.wav": (CLIPS / "sentence.wav").read_bytes()}
    body = "\n".join(files)
    clips, notes = fa.load_attachments(body, fetch=lambda u: files[u])
    assert [k for _, k, _ in clips] == ["positive", "negative", None] and notes == []
    res = fa.score_clips(clips, ROOT / "models" / "kalki_int8.onnx")
    rows, _, n = res
    assert n == {"positive": 1, "negative": 1}
    r = next(r for r in rows if r["threshold"] == 0.8)
    assert r["caught"] == 1 and r["false_alarms"] == 0
    md = fa.report(clips, notes, res)
    assert "1/1" in md and "no label and were skipped" in md and "deleted after scoring" in md


def test_unlabelled_or_missing_audio_message():
    md = fa.report([], ["could not download `x.wav` (boom)"], None)
    assert "could not find labelled audio" in md.lower() and "boom" in md


def test_install_writes_numbered_training_clips(tmp_path):
    x = np.random.default_rng(1).standard_normal(16000).astype(np.float32) * 0.1
    z = tmp_path / "c.zip"
    with zipfile.ZipFile(z, "w") as zf:
        zf.writestr("positive/a.wav", wav_bytes(x))
        zf.writestr("positive/b.wav", wav_bytes(x))
        zf.writestr("negative/c.wav", wav_bytes(x))
        zf.writestr("d.wav", wav_bytes(x))  # unlabelled: left out
    from types import SimpleNamespace
    data = tmp_path / "data"
    fa.cmd_install(SimpleNamespace(zip=str(z), data=str(data)))
    fa.cmd_install(SimpleNamespace(zip=str(z), data=str(data)))
    assert sorted(p.name for p in (data / "positive").iterdir()) == [f"pos_real_{i}.wav" for i in range(4)]
    assert sorted(p.name for p in (data / "negative").iterdir()) == ["neg_real_0.wav", "neg_real_1.wav"]
