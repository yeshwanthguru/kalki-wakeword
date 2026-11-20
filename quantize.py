"""Make smaller copies of the wake word model (fp16 and int8) and check they still detect.

    python quantize.py
Writes single-file ONNX models to wakeword/models/ and prints accuracy next to the fp32 original.
"""
from pathlib import Path

import numpy as np
import onnx
import onnxruntime as ort
import torch
from onnxconverter_common import float16
from onnx import TensorProto, helper, numpy_helper

from benchmark import roc_metrics, splits
from shrink_model import build_int4
from model import WIN, WakeNet
from train import Sampler

HERE = Path(__file__).parent
MODELS = HERE / "models"


def single_file(src, dst):
    onnx.save(onnx.load(str(src)), str(dst), save_as_external_data=False)


def weight_only_int8(src, dst, min_size=500):
    """Store big float weights as int8 + per-channel scale; a DequantizeLinear node restores them at load."""
    m = onnx.load(str(src))
    new_inits, nodes = [], []
    for init in m.graph.initializer:
        w = numpy_helper.to_array(init)
        if w.dtype != np.float32 or w.size < min_size:
            new_inits.append(init)
            continue
        axes = tuple(range(1, w.ndim))
        scale = np.maximum(np.abs(w).max(axis=axes), 1e-12) / 127.0 if axes else np.array(max(np.abs(w).max(), 1e-12) / 127.0)
        shape = (-1,) + (1,) * (w.ndim - 1)
        q = np.clip(np.round(w / scale.reshape(shape)), -127, 127).astype(np.int8)
        new_inits += [numpy_helper.from_array(q, init.name + "_q"),
                      numpy_helper.from_array(scale.astype(np.float32), init.name + "_scale")]
        nodes.append(helper.make_node("DequantizeLinear", [init.name + "_q", init.name + "_scale"], [init.name], axis=0))
    del m.graph.initializer[:]
    m.graph.initializer.extend(new_inits)
    nodes += list(m.graph.node)
    del m.graph.node[:]
    m.graph.node.extend(nodes)
    onnx.save(m, str(dst))


def probs(path, x):
    sess = ort.InferenceSession(str(path), providers=["CPUExecutionProvider"])
    out = []
    for i in range(len(x)):  # exported GRU is fixed at batch 1, which is how live.py runs it
        out.append(sess.run(None, {"audio": x[i:i + 1]})[0])
    return 1 / (1 + np.exp(-np.concatenate(out).astype(np.float32)))


def main():
    MODELS.mkdir(exist_ok=True)
    m = WakeNet(); m.load_state_dict(torch.load(HERE / "out" / "kalki.pt", map_location="cpu")); m.eval()
    tmp = HERE / "out" / "fp32_tmp.onnx"
    torch.onnx.export(m, torch.zeros(1, WIN), tmp, input_names=["audio"], output_names=["logit"], opset_version=17)
    fp32 = MODELS / "kalki_fp32.onnx"
    single_file(tmp, fp32)

    fp16 = MODELS / "kalki_fp16.onnx"
    onnx.save(float16.convert_float_to_float16(onnx.load(str(fp32)), keep_io_types=True), str(fp16))

    int8 = MODELS / "kalki_int8.onnx"
    weight_only_int8(fp32, int8)

    int4 = MODELS / "kalki_int4.onnx"
    build_int4(fp32, int4)    # 4-bit weights and a computed front end: 36 KB (see shrink_model.py)

    _, _, vp, vn = splits()
    x, y = Sampler(vp, vn, 123).batch(4000, "cpu")
    x, y = x.numpy(), y.numpy() > 0.5
    base = probs(fp32, x)
    print(f"{'model':<8} {'size KB':>8} {'AUC':>7} {'recall@.5':>10} {'fpr@.5':>8} {'max |dp| vs fp32':>17}")
    for name, path in [("fp32", fp32), ("fp16", fp16), ("int8", int8), ("int4", int4)]:
        p = probs(path, x)
        r, _ = roc_metrics(p[y], p[~y])
        print(f"{name:<8} {path.stat().st_size / 1024:8.0f} {r['auc']:7.4f} {r['recall@0.5']:10.3f} "
              f"{r['fpr@0.5']:8.3f} {np.abs(p - base).max():17.4f}")


if __name__ == "__main__":
    main()
