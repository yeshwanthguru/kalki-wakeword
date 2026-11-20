import assert from "node:assert/strict";
import { readFile } from "node:fs/promises";
import { test } from "node:test";
import * as ort from "onnxruntime-node";
import { HOP, SENSITIVITY, SR, WakeWordDetector, defaultModelUrl, fallbackModelUrl, loadModelBytes } from "../index.js";

const CLIPS = new URL("../../docs/clips/", import.meta.url);

/** 16-bit PCM mono wav -> Float32Array in [-1, 1]. */
function readWav(buf) {
  const dv = new DataView(buf.buffer, buf.byteOffset, buf.byteLength);
  let pos = 12, fmt = null;
  while (pos + 8 <= buf.length) {
    const id = String.fromCharCode(...buf.subarray(pos, pos + 4)), size = dv.getUint32(pos + 4, true);
    if (id === "fmt ") fmt = { channels: dv.getUint16(pos + 10, true), rate: dv.getUint32(pos + 12, true), bits: dv.getUint16(pos + 22, true) };
    if (id === "data") {
      assert.deepEqual(fmt, { channels: 1, rate: SR, bits: 16 });
      const n = size / 2, out = new Float32Array(n);
      for (let i = 0; i < n; i++) out[i] = dv.getInt16(pos + 8 + 2 * i, true) / 32768;
      return out;
    }
    pos += 8 + size + (size % 2);
  }
  throw new Error("no data chunk");
}

async function run(name, chunk = HOP, options = {}) {
  const x = readWav(new Uint8Array(await readFile(new URL(`${name}.wav`, CLIPS))));
  const det = await WakeWordDetector.create(ort, options);
  const hits = [];
  for (let i = 0; i < x.length; i += chunk) hits.push(...(await det.push(x.subarray(i, i + chunk))));
  hits.push(...(await det.push(new Float32Array(4000)))); // let the last window finish
  return hits;
}

// Reference values come from the Python package (kalki_wakeword, default model) on the same clips.
const FIRES = { "hey-kalki-1": [1.6, 1.0], "hey-kalki-2": [1.52, 0.9994], "hey-kalki-3": [1.52, 0.9989] };

for (const [name, [time, prob]] of Object.entries(FIRES)) {
  test(`fires once on ${name}, at the same time and probability as Python`, async () => {
    const hits = await run(name);
    assert.equal(hits.length, 1);
    assert.ok(Math.abs(hits[0].time - time) < 0.011, `time ${hits[0].time} vs ${time}`);
    assert.ok(Math.abs(hits[0].prob - prob) < 5e-4, `prob ${hits[0].prob} vs ${prob}`);
  });
}

for (const name of ["hey-kali", "hey-google", "kalki-alone", "sentence"]) {
  test(`stays quiet on ${name}`, async () => assert.equal((await run(name)).length, 0));
}

test("the chunk size does not change the result", async () => {
  const a = await run("hey-kalki-1", 1280), b = await run("hey-kalki-1", 333), c = await run("hey-kalki-1", 5000);
  assert.deepEqual(b.map((h) => h.time), a.map((h) => h.time));
  assert.deepEqual(c.map((h) => h.time), a.map((h) => h.time));
});

test("overlapping push calls are processed in order", async () => {
  const x = readWav(new Uint8Array(await readFile(new URL("hey-kalki-1.wav", CLIPS))));
  const det = await WakeWordDetector.create(ort);
  const parts = [];
  for (let i = 0; i < x.length; i += 2000) parts.push(det.push(x.subarray(i, i + 2000))); // not awaited one by one
  parts.push(det.push(new Float32Array(4000)));
  assert.equal((await Promise.all(parts)).flat().length, 1);
});

test("settings: a higher threshold or more hits can suppress the detection", async () => {
  assert.equal((await run("hey-kalki-1", HOP, { threshold: 0.99999 })).length, 0);
  assert.equal((await run("hey-kalki-1", HOP, { hits: 50 })).length, 0);
});

test("loads the model from bytes and reports a missing file clearly", async () => {
  const bytes = await loadModelBytes();
  assert.ok(bytes.length > 30_000 && bytes.length < 50_000); // the default is the 36 KB model
  const det = await WakeWordDetector.create(ort, { model: bytes });
  assert.equal((await det.push(new Float32Array(HOP))).length, 0);
  await assert.rejects(loadModelBytes(new URL("file:///no/such/model.onnx")));
});

test("sensitivity presets: the default is balanced, overrides win, and every preset still detects", async () => {
  assert.deepEqual(SENSITIVITY, { sensitive: [0.6, 2], balanced: [0.8, 2], strict: [0.9, 3] });
  const d = await WakeWordDetector.create(ort);
  assert.deepEqual([d.threshold, d.hits], [0.8, 2]);
  const s = await WakeWordDetector.create(ort, { sensitivity: "strict", threshold: 0.5 });
  assert.deepEqual([s.threshold, s.hits], [0.5, 3]);
  await assert.rejects(() => WakeWordDetector.create(ort, { sensitivity: "paranoid" }), /sensitivity must be one of/);
  for (const sensitivity of Object.keys(SENSITIVITY)) {
    assert.equal((await run("hey-kalki-1", HOP, { sensitivity })).length, 1, sensitivity);
    assert.equal((await run("hey-google", HOP, { sensitivity })).length, 0, sensitivity);
  }
});

test("the default model is the 36 KB one, and the 232 KB one is the fallback", async () => {
  assert.ok((await loadModelBytes(defaultModelUrl)).length < 50_000);
  assert.ok((await loadModelBytes(fallbackModelUrl)).length > 200_000);
  assert.equal((await WakeWordDetector.create(ort)).modelSource, defaultModelUrl);

  // a runtime that cannot read 4-bit weights fails on the first model; the detector must fall back and still work
  let calls = 0;
  const oldRuntime = { ...ort, InferenceSession: { create: async (bytes, opts) => {
    if (calls++ === 0) throw new Error("unsupported type int4");
    return ort.InferenceSession.create(bytes, opts);
  } } };
  const det = await WakeWordDetector.create({ Tensor: ort.Tensor, InferenceSession: oldRuntime.InferenceSession });
  assert.equal(det.modelSource, fallbackModelUrl);
  assert.equal(calls, 2);
});
