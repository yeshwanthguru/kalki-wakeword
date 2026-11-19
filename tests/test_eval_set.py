import json
import zipfile
from pathlib import Path

import numpy as np
import pytest
import soundfile as sf

import eval_set


def make_set(folder, voices=("v1", "v2")):
    folder = Path(folder)
    files = {}
    import hashlib
    for sub, tag in (("positive", "pos"), ("negative", "hard")):
        (folder / sub).mkdir(parents=True, exist_ok=True)
        for v in voices:
            p = folder / sub / f"{tag}_{v}_0.wav"
            sf.write(p, np.full(8000, 0.1, np.float32), 16000)
            files[f"{sub}/{p.name}"] = hashlib.sha256(p.read_bytes()).hexdigest()
    (folder / "MANIFEST.json").write_text(json.dumps({"version": "vtest", "voices": list(voices), "n_files": len(files), "files": files}))


def test_read_side(tmp_path):
    assert not eval_set.exists(tmp_path) and eval_set.version(tmp_path) is None and eval_set.voices(tmp_path) == set()
    make_set(tmp_path)
    assert eval_set.exists(tmp_path) and eval_set.version(tmp_path) == "vtest" and eval_set.voices(tmp_path) == {"v1", "v2"}
    pos, neg = eval_set.load_set(tmp_path)
    assert sorted(p.name for p in pos) == ["pos_v1_0.wav", "pos_v2_0.wav"] and len(neg) == 2
    eval_set.verify(tmp_path)


def test_stream_minutes_defaults_and_reads_the_manifest(tmp_path):
    assert eval_set.stream_minutes(tmp_path) == 15            # no set: the original stream length
    make_set(tmp_path)
    assert eval_set.stream_minutes(tmp_path) == 15            # a v1-style manifest without the field
    m = json.loads((tmp_path / "MANIFEST.json").read_text())
    m["stream_minutes"] = 60
    (tmp_path / "MANIFEST.json").write_text(json.dumps(m))
    assert eval_set.stream_minutes(tmp_path) == 60


def test_verify_catches_a_changed_clip(tmp_path):
    make_set(tmp_path)
    sf.write(tmp_path / "positive" / "pos_v1_0.wav", np.zeros(8000, np.float32), 16000)
    with pytest.raises(SystemExit):
        eval_set.verify(tmp_path)


def test_unpack_roundtrip_and_ignores_path_traversal(tmp_path):
    src = tmp_path / "src"; make_set(src)
    z = tmp_path / "set.zip"
    with zipfile.ZipFile(z, "w") as zf:
        for f in src.rglob("*"):
            if f.is_file():
                zf.write(f, f.relative_to(src))
        zf.writestr("../../escaped.txt", "x")
    dest = tmp_path / "out" / "eval_set"
    m = eval_set.unpack(z, dest)
    assert m["version"] == "vtest" and (dest / "positive" / "pos_v1_0.wav").exists()
    assert not (tmp_path / "escaped.txt").exists() and not (tmp_path / "out" / "escaped.txt").exists()


torch = pytest.importorskip("torch")


def test_split_data_uses_frozen_set_and_excludes_its_voices(tmp_path, monkeypatch):
    import train
    make_set(tmp_path, voices=("v1",))
    monkeypatch.setattr(eval_set, "EVAL_DIR", tmp_path)
    for name in ("exists", "voices", "load_set"):
        orig = getattr(eval_set, name)
        monkeypatch.setattr(eval_set, name, lambda *a, _o=orig, **k: _o(tmp_path))
    pos = {Path(f"pos_{v}_0.wav"): np.zeros(10, np.float32) for v in ("v1", "v2", "real")}
    neg = {Path(f"hard_{v}_0.wav"): np.zeros(10, np.float32) for v in ("v1", "v3")}
    tp, tn, vp, vn = train.split_data(pos, neg)
    assert sorted(p.name for p in tp) == ["pos_real_0.wav", "pos_v2_0.wav"] and sorted(p.name for p in tn) == ["hard_v3_0.wav"]
    assert [p.name for p in vp] == ["pos_v1_0.wav"] and all(p.parent == tmp_path / "positive" for p in vp)


def test_report_does_not_compare_against_a_baseline_from_another_set():
    import ci_eval
    row = {"auc": 0.99, "recall@0.5": 0.9, "fpr@0.5": 0.02, "stream": {"0.8": {"detected": 0.9, "false_alarms_per_hour": 10.0}}}
    cur = {"fp32": row, "eval_set": "v1"}
    same = ci_eval.report(cur, {"fp32": {**row, "auc": 0.999}, "eval_set": "v1"}, 0.005, 8)
    assert same[1] and not same[2]                       # a real drop on the same set fails
    other = ci_eval.report(cur, {"fp32": {**row, "auc": 0.999}, "eval_set": "regenerated"}, 0.005, 8)
    assert other[1] == [] and other[2]                   # different set: not compared, flagged stale


def test_heldout_fraction_holds_out_more_voices():
    import train
    pos = {Path(f"pos_voice{i:02d}_0.wav"): 0 for i in range(40)}
    assert len(train.heldout_voices(pos)) == 5 and len(train.heldout_voices(pos, 4)) == 10
    assert train.heldout_voices(pos, 4) >= set()                       # deterministic, same call twice
    assert train.heldout_voices(pos, 4) == train.heldout_voices(pos, 4)
