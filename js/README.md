# kalki-wakeword

A 36 KB "hey kalki" wake word model and streaming detector for the browser and Node.js. It runs on the CPU (WebAssembly in a
browser) and has no dependencies of its own: you bring the ONNX runtime, `onnxruntime-web` or `onnxruntime-node`.

```js
import * as ort from "onnxruntime-web";               // "onnxruntime-node" in Node
import { WakeWordDetector } from "kalki-wakeword";

const det = await WakeWordDetector.create(ort, { threshold: 0.8 });   // loads the bundled int8 model
// samples: Float32Array of mono 16 kHz audio in [-1, 1], any chunk size
for (const hit of await det.push(samples)) {
  console.log(`heard it at ${hit.time.toFixed(2)}s (p=${hit.prob.toFixed(2)})`);
}
```

- `WakeWordDetector.create(ort, { model, threshold = 0.8, hits = 2, cooldown = 1.5, sessionOptions })`.
  `model` is a URL, `file://` URL, `ArrayBuffer` or `Uint8Array`. The default is the bundled 36 KB model (4-bit weights, needs
  onnxruntime 1.20 or newer) and, if the runtime cannot load it, the bundled 232 KB int8 model.
- `push()` is async and processes calls in order. `reset()` clears the stream state.
- In a browser, your bundler or server must serve `models/kalki_int4.onnx` and `models/kalki_int8.onnx` (the default model URLs are relative to the module) and
  the `onnxruntime-web` WebAssembly files.
- The same trigger rule and model as the Python package; the tests check that detection times and probabilities match it.

Source, training code, benchmarks and the live demo: https://github.com/yeshwanthguru/kalki-wakeword. The model was trained on
synthetic speech only, so expect more false alarms than a commercial detector; the project README has the numbers.
