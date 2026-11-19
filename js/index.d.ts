export const SR: 16000;
export const WIN: 24000;
export const HOP: 1280;
export const defaultModelUrl: URL;

export interface Detection {
  /** seconds of audio fed so far when it fired */
  time: number;
  prob: number;
}

export interface DetectorSettings {
  /** probability a window must exceed (default 0.8) */
  threshold?: number;
  /** consecutive windows above the threshold needed to fire (default 2) */
  hits?: number;
  /** seconds to wait after a detection before the next one (default 1.5) */
  cooldown?: number;
}

export interface CreateOptions extends DetectorSettings {
  /** URL, file:// URL, ArrayBuffer or Uint8Array of the ONNX model (default: the bundled int8 model) */
  model?: URL | string | ArrayBuffer | Uint8Array;
  /** passed to InferenceSession.create */
  sessionOptions?: object;
}

export function loadModelBytes(source?: URL | string | ArrayBuffer | Uint8Array): Promise<Uint8Array>;

export class WakeWordDetector {
  constructor(ort: unknown, session: unknown, settings?: DetectorSettings);
  static create(ort: unknown, options?: CreateOptions): Promise<WakeWordDetector>;
  threshold: number;
  hits: number;
  cooldown: number;
  lastProb: number;
  reset(): void;
  prob(window: Float32Array): Promise<number>;
  /** mono 16 kHz audio in [-1, 1], any chunk size */
  push(samples: Float32Array): Promise<Detection[]>;
}
