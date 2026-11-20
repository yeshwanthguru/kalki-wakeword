// "hey kalki" wake word detector for the browser and Node.js.
//
//   import * as ort from "onnxruntime-web";            // in a browser (or "onnxruntime-node" in Node)
//   import { WakeWordDetector } from "kalki-wakeword";
//   const det = await WakeWordDetector.create(ort);    // loads the bundled 36 KB model (falls back to a 232 KB one)
//   for (const hit of await det.push(samples)) console.log(`heard it at ${hit.time.toFixed(2)}s (p=${hit.prob.toFixed(2)})`);
//
// `samples` is a Float32Array of mono 16 kHz audio in [-1, 1], any chunk size. You bring the ONNX runtime, so the package has no
// hard dependency and works with either runtime. Same trigger rule as the Python package and the browser demo.

export const SR = 16000;
export const WIN = 24000; // model input: 1.5 s
export const HOP = 1280; // scored every 80 ms

// Named trade-offs between missed wake words and false alarms: [probability threshold, consecutive windows that must exceed it].
// Measured for the shipped model on eval-set-v2 (see the Python package's SENSITIVITY): sensitive about 98% caught / 44 false alarms
// per hour, balanced (default) about 97% / 27, strict about 95% / 13. A starting point: confirm on your own recordings.
export const SENSITIVITY = { sensitive: [0.6, 2], balanced: [0.8, 2], strict: [0.9, 3] };

export const defaultModelUrl = new URL("./models/kalki_int4.onnx", import.meta.url); // 36 KB, needs onnxruntime >= 1.20
export const fallbackModelUrl = new URL("./models/kalki_int8.onnx", import.meta.url); // 232 KB, loads on any version

/** Model bytes from a URL (http(s):// or file://), a path-like string, an ArrayBuffer or a Uint8Array. */
export async function loadModelBytes(source = defaultModelUrl) {
  if (source instanceof Uint8Array) return source;
  if (source instanceof ArrayBuffer) return new Uint8Array(source);
  const url = source instanceof URL ? source : new URL(String(source), import.meta.url);
  if (url.protocol === "file:") {
    const { readFile } = await import("node:fs/promises"); // only reached in Node
    return new Uint8Array(await readFile(url));
  }
  const res = await fetch(url);
  if (!res.ok) throw new Error(`could not load the model from ${url}: HTTP ${res.status}`);
  return new Uint8Array(await res.arrayBuffer());
}

export class WakeWordDetector {
  /** Prefer WakeWordDetector.create(ort, options), which loads the model for you. */
  constructor(ort, session, { threshold, hits, cooldown = 1.5, sensitivity = "balanced" } = {}) {
    if (!Object.hasOwn(SENSITIVITY, sensitivity)) {
      throw new Error(`sensitivity must be one of ${Object.keys(SENSITIVITY).join(", ")}, got ${sensitivity}`);
    }
    this.ort = ort;
    this.session = session;
    this.threshold = threshold ?? SENSITIVITY[sensitivity][0];
    this.hits = hits ?? SENSITIVITY[sensitivity][1];
    this.cooldown = cooldown;
    this.reset();
  }

  /**
   * @param ort the module `onnxruntime-web` or `onnxruntime-node`
   * @param options { model (default: the bundled 36 KB model, with a fallback), sensitivity = "balanced" | "sensitive" | "strict", threshold, hits, cooldown = 1.5, sessionOptions }
   *   threshold and hits override the sensitivity preset (default balanced: 0.8 and 2)
   */
  static async create(ort, { model, sessionOptions = {}, ...settings } = {}) {
    const open = async (source) => ort.InferenceSession.create(await loadModelBytes(source), sessionOptions);
    let session, used = model;
    if (model === undefined) {
      // default: the 36 KB model, or the 232 KB one if this onnxruntime cannot load 4-bit weights (it needs 1.20 or newer)
      try { session = await open((used = defaultModelUrl)); } catch { session = await open((used = fallbackModelUrl)); }
    } else {
      session = await open(model);
    }
    const detector = new WakeWordDetector(ort, session, settings);
    detector.modelSource = used; // which model was loaded
    return detector;
  }

  reset() {
    this.buf = new Float32Array(WIN);
    this.pending = new Float32Array(0);
    this.streak = 0;
    this.clock = 0;
    this.lastFire = -1e9;
    this.lastProb = 0;
    this._busy = Promise.resolve();
  }

  async prob(window) {
    const out = await this.session.run({ audio: new this.ort.Tensor("float32", window, [1, WIN]) });
    return 1 / (1 + Math.exp(-out[Object.keys(out)[0]].data[0]));
  }

  /** Feed mono 16 kHz audio (Float32Array in [-1, 1]). Resolves to the detections ({ time, prob }) that fired in this chunk. */
  push(samples) {
    const run = this._busy.then(() => this._push(samples));
    this._busy = run.catch(() => {}); // calls are processed in order, even if one of them fails
    return run;
  }

  async _push(samples) {
    const joined = new Float32Array(this.pending.length + samples.length);
    joined.set(this.pending);
    joined.set(samples, this.pending.length);
    const fired = [];
    let at = 0;
    for (; joined.length - at >= HOP; at += HOP) {
      this.buf.copyWithin(0, HOP);
      this.buf.set(joined.subarray(at, at + HOP), WIN - HOP);
      this.lastProb = await this.prob(this.buf);
      this.clock += HOP / SR;
      this.streak = this.lastProb > this.threshold ? this.streak + 1 : 0;
      if (this.streak >= this.hits && this.clock - this.lastFire > this.cooldown) {
        this.lastFire = this.clock;
        fired.push({ time: this.clock, prob: this.lastProb });
      }
    }
    this.pending = joined.slice(at);
    return fired;
  }
}
