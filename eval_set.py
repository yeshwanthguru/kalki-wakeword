"""A frozen evaluation set, so scores do not drift when the TTS clips are regenerated.

    python eval_set.py build --out eval_set --zip eval-set.zip --version v1   # after gen_tts.py; run once
    python eval_set.py fetch                                                  # download the published set (gh)
Regenerating clips with edge-tts gives slightly different audio every time, which moved the PR gate by +/-8 false alarms per
hour with identical models. The published set (GitHub release 'eval-set-<version>') holds the held-out voices' clips once and
for all. When eval_set/ exists, train.split_data() uses it as the held-out data and keeps those voices out of training, and
every benchmark, ci_eval.py and the PR gate score against exactly the same audio.
"""
import argparse
import hashlib
import json
import shutil
import subprocess
import zipfile
from pathlib import Path

import numpy as np
import soundfile as sf

HERE = Path(__file__).parent
EVAL_DIR = HERE / "eval_set"


def exists(folder=EVAL_DIR):
    return (Path(folder) / "MANIFEST.json").exists()


def manifest(folder=EVAL_DIR):
    return json.loads((Path(folder) / "MANIFEST.json").read_text())


def version(folder=EVAL_DIR):
    return manifest(folder)["version"] if exists(folder) else None


def load_set(folder=EVAL_DIR):
    """(positives, negatives) as {Path: samples}, the same shape train.load() returns."""
    out = []
    for sub in ("positive", "negative"):
        out.append({p: sf.read(p, dtype="float32")[0] for p in sorted((Path(folder) / sub).glob("*.wav"))})
    return tuple(out)


def voices(folder=EVAL_DIR):
    return set(manifest(folder)["voices"]) if exists(folder) else set()


def build(out, zip_path=None, version="v1"):
    """Copy the held-out voices' clips from data/ into out/ with a manifest of checksums."""
    from train import heldout_voices, load, voice_of   # torch is only needed for building, not for reading
    held = heldout_voices(load("positive"))
    out = Path(out)
    if out.exists():
        shutil.rmtree(out)
    files = {}
    for sub in ("positive", "negative"):
        (out / sub).mkdir(parents=True)
        for p in sorted((HERE / "data" / sub).glob("*.wav")):
            if voice_of(p) in held:
                shutil.copy(p, out / sub / p.name)
                files[f"{sub}/{p.name}"] = hashlib.sha256(p.read_bytes()).hexdigest()
    m = {"version": version, "voices": sorted(held), "n_files": len(files), "files": files}
    (out / "MANIFEST.json").write_text(json.dumps(m, indent=1))
    if zip_path:
        with zipfile.ZipFile(zip_path, "w", zipfile.ZIP_DEFLATED) as z:
            for f in sorted(out.rglob("*")):
                if f.is_file():
                    z.write(f, f.relative_to(out))
    return m


def verify(folder=EVAL_DIR):
    """Check every file against the manifest (a corrupted or tampered set must not silently change scores)."""
    m = manifest(folder)
    bad = [k for k, h in m["files"].items()
           if not (Path(folder) / k).exists() or hashlib.sha256((Path(folder) / k).read_bytes()).hexdigest() != h]
    if bad:
        raise SystemExit(f"eval set does not match its manifest: {bad[:3]}")
    return m


def unpack(zip_path, folder=EVAL_DIR):
    folder = Path(folder)
    if folder.exists():
        shutil.rmtree(folder)
    folder.mkdir(parents=True)
    with zipfile.ZipFile(zip_path) as z:
        for i in z.infolist():   # names are only used after a path check, never trusted
            target = (folder / i.filename).resolve()
            if i.is_dir() or folder.resolve() not in target.parents:
                continue
            target.parent.mkdir(parents=True, exist_ok=True)
            target.write_bytes(z.read(i))
    return verify(folder)


def cmd_fetch(a):
    tag = f"eval-set-{a.version}"
    tmp = HERE / "eval-set.zip"
    r = subprocess.run(["gh", "release", "download", tag, "-p", "eval-set.zip", "-O", str(tmp), "--clobber"],
                       capture_output=True, text=True)
    if r.returncode:
        print(f"no published eval set '{tag}' ({r.stderr.strip()[:200]}); scores will use regenerated clips")
        return
    m = unpack(tmp)
    tmp.unlink()
    print(f"eval set {m['version']}: {m['n_files']} clips from {len(m['voices'])} held-out voices")


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest="cmd", required=True)
    b = sub.add_parser("build")
    b.add_argument("--out", default=str(EVAL_DIR))
    b.add_argument("--zip")
    b.add_argument("--version", default="v1")
    b.set_defaults(fn=lambda a: print(json.dumps({k: v for k, v in build(a.out, a.zip, a.version).items() if k != "files"})))
    f = sub.add_parser("fetch")
    f.add_argument("--version", default="v1")
    f.set_defaults(fn=cmd_fetch)
    a = ap.parse_args()
    a.fn(a)
