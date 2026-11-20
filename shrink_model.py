"""Build the 36 KB model: kalki_int4.onnx, from the fp32 model, with no retraining.

    python shrink_model.py [--src models/kalki_fp32.onnx] [--out models/kalki_int4.onnx]

Two changes make the file about 6x smaller than kalki_int8.onnx with the same accuracy:
 1. 72% of the int8 file was the audio front end's fixed tables (cosine and sine for the Fourier transform, and the mel filter
    bank). They are plain math, so the model now computes them with a few built-in operations when it loads (onnxruntime
    constant-folds them once; speed and outputs are unchanged to within 0.0002 in the logit) instead of storing them.
 2. The ~54,000 learned weights are stored as 4-bit integers (two per byte, one scale per output channel). This network is very
    tolerant of that: AUC on eval-set-v2 is unchanged.
The result needs onnxruntime 1.20 or newer (4-bit DequantizeLinear, ONNX opset 21); the packages fall back to kalki_int8.onnx
when the model cannot be loaded. The output is deterministic, and a test checks that models/kalki_int4.onnx is exactly what this
script produces from models/kalki_fp32.onnx.
"""
import argparse
import math
import os
from pathlib import Path

import numpy as np
import onnx
from onnx import TensorProto as T
from onnx import helper, numpy_helper

HERE = Path(__file__).parent
# These must match model.py (a test checks that they do); they are repeated here so this script needs no torch.
SR, N_FFT, N_MELS, FMIN, FMAX = 16000, 400, 40, 60.0, 7600.0
BITS = 4
LEVELS = 2 ** (BITS - 1) - 1            # symmetric: -7..7
MIN_WEIGHT_SIZE = 500                   # smaller tensors (biases, norms, the output layer) stay float32
OPSET = 21                              # 4-bit DequantizeLinear
FRONT_END = ("mel.cos", "mel.sin", "mel.fb")


def _mel_edges():
    """Lower, centre and upper edge (Hz) of each mel triangle: 3 x 40 numbers instead of a 40 x 201 table."""
    hz2mel = lambda f: 2595 * math.log10(1 + f / 700)
    mel2hz = lambda m: 700 * (10 ** (m / 2595) - 1)
    pts = np.array([mel2hz(hz2mel(FMIN) + i * (hz2mel(FMAX) - hz2mel(FMIN)) / (N_MELS + 1)) for i in range(N_MELS + 2)], np.float32)
    return pts[:-2, None], pts[1:-1, None], pts[2:, None]


def _front_end_graph():
    """Initializers and nodes that produce mel.cos, mel.sin and mel.fb (same values as the stored tables)."""
    bins = N_FFT // 2 + 1
    lo, centre, hi = _mel_edges()
    f32 = lambda name, v: numpy_helper.from_array(np.asarray(v, np.float32), name)
    i64 = lambda name, v: numpy_helper.from_array(np.asarray(v, np.int64), name)
    inits = [f32("fe_zero", 0.0), f32("fe_one", 1.0), f32("fe_nfft", float(N_FFT)), f32("fe_nbins", float(bins)),
             f32("fe_two_pi_over_n", 2 * math.pi / N_FFT), f32("fe_half", 0.5), f32("fe_neg_one", -1.0),
             f32("fe_bin_hz", (SR / 2) / (bins - 1)), f32("fe_lo", lo), f32("fe_c", centre), f32("fe_hi", hi),
             i64("fe_axis1", [1]), i64("fe_axis0", [0])]
    N = helper.make_node
    nodes = [
        N("Range", ["fe_zero", "fe_nfft", "fe_one"], ["fe_n"]),                    # sample index 0..399
        N("Range", ["fe_zero", "fe_nbins", "fe_one"], ["fe_k"]),                   # frequency bin 0..200
        N("Unsqueeze", ["fe_k", "fe_axis1"], ["fe_k_col"]), N("Unsqueeze", ["fe_n", "fe_axis0"], ["fe_n_row"]),
        N("Mul", ["fe_k_col", "fe_n_row"], ["fe_kn"]), N("Mul", ["fe_kn", "fe_two_pi_over_n"], ["fe_angle"]),
        N("Cos", ["fe_angle"], ["fe_cos"]), N("Sin", ["fe_angle"], ["fe_sin"]),
        N("Mul", ["fe_n", "fe_two_pi_over_n"], ["fe_nang"]), N("Cos", ["fe_nang"], ["fe_ncos"]),   # periodic Hann window
        N("Mul", ["fe_ncos", "fe_half"], ["fe_hc"]), N("Sub", ["fe_half", "fe_hc"], ["fe_win"]),
        N("Mul", ["fe_cos", "fe_win"], ["fe_cosw"]), N("Mul", ["fe_sin", "fe_win"], ["fe_sinw"]),
        N("Mul", ["fe_sinw", "fe_neg_one"], ["fe_sinw_neg"]),
        N("Unsqueeze", ["fe_cosw", "fe_axis1"], ["mel.cos"]), N("Unsqueeze", ["fe_sinw_neg", "fe_axis1"], ["mel.sin"]),
        N("Mul", ["fe_k", "fe_bin_hz"], ["fe_hz"]),
        N("Sub", ["fe_hz", "fe_lo"], ["fe_up_n"]), N("Sub", ["fe_c", "fe_lo"], ["fe_up_d"]), N("Div", ["fe_up_n", "fe_up_d"], ["fe_up"]),
        N("Sub", ["fe_hi", "fe_hz"], ["fe_dn_n"]), N("Sub", ["fe_hi", "fe_c"], ["fe_dn_d"]), N("Div", ["fe_dn_n", "fe_dn_d"], ["fe_dn"]),
        N("Min", ["fe_up", "fe_dn"], ["fe_tri"]), N("Max", ["fe_tri", "fe_zero"], ["mel.fb"]),
    ]
    return inits, nodes


def _pack_int4(q):
    """int8 values in -8..7 -> bytes, two per byte, first value in the low nibble (the ONNX INT4 layout)."""
    flat = q.reshape(-1).astype(np.int8)
    if len(flat) % 2:
        flat = np.append(flat, np.int8(0))
    nib = (flat & 0x0F).astype(np.uint8)
    return (nib[0::2] | (nib[1::2] << 4)).astype(np.uint8).tobytes()


def _shorten_names(m):
    """Short tensor names and no doc strings or metadata: about 5 KB of a file this small."""
    keep, ren = {"audio", "logit"}, {}
    short = lambda n: n if n in keep or n == "" else ren.setdefault(n, f"t{len(ren)}")
    for i in m.graph.initializer:
        i.name = short(i.name)
    for n in m.graph.node:
        n.name, n.doc_string = "", ""
        for k in range(len(n.input)):
            n.input[k] = short(n.input[k])
        for k in range(len(n.output)):
            n.output[k] = short(n.output[k])
    m.graph.name, m.doc_string, m.producer_name, m.producer_version = "g", "", "", ""
    del m.metadata_props[:], m.graph.value_info[:]


def build_int4(src, dst):
    """Write the 4-bit model with the computed front end to dst. Returns the file size in bytes."""
    m = onnx.load(str(src))
    names = {i.name for i in m.graph.initializer}
    missing = [n for n in FRONT_END if n not in names]
    if missing:
        raise SystemExit(f"{src} has no stored front-end tables {missing}: pass the fp32 model exported by train.py/quantize.py")
    fe_inits, fe_nodes = _front_end_graph()
    inits, deq = [], []
    for i in m.graph.initializer:
        if i.name in FRONT_END:
            continue
        a = numpy_helper.to_array(i)
        if a.dtype != np.float32 or a.size < MIN_WEIGHT_SIZE:
            inits.append(i)
            continue
        w = a.reshape(a.shape[0], -1)
        scale = (np.maximum(np.abs(w).max(1), 1e-12) / LEVELS).astype(np.float32)
        q = np.clip(np.round(w / scale[:, None]), -LEVELS, LEVELS).astype(np.int8).reshape(a.shape)
        inits += [helper.make_tensor(i.name + "_q", T.INT4, list(a.shape), _pack_int4(q), raw=True),
                  numpy_helper.from_array(scale, i.name + "_s")]
        deq.append(helper.make_node("DequantizeLinear", [i.name + "_q", i.name + "_s"], [i.name], axis=0))
    nodes = fe_nodes + deq + list(m.graph.node)
    del m.graph.initializer[:], m.graph.node[:]
    m.graph.initializer.extend(inits + fe_inits)
    m.graph.node.extend(nodes)
    for op in m.opset_import:
        if op.domain in ("", "ai.onnx"):
            op.version = OPSET
    m.ir_version = max(m.ir_version, 10)
    _shorten_names(m)
    onnx.checker.check_model(m)
    Path(dst).parent.mkdir(parents=True, exist_ok=True)
    onnx.save(m, str(dst))
    return os.path.getsize(dst)


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--src", default=str(HERE / "models" / "kalki_fp32.onnx"))
    ap.add_argument("--out", default=str(HERE / "models" / "kalki_int4.onnx"))
    a = ap.parse_args()
    print(f"wrote {a.out}: {build_int4(a.src, a.out)} bytes")
