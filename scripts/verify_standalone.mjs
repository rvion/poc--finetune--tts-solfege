// Vérifie que le forward CNN JS pur (web/nn.js + web/features.js) reproduit
// EXACTEMENT les prédictions PyTorch. Charge les cas de référence produits par
// scripts/dump_verify_cases.py (waveforms + probas attendues) et compare.
// Sort en code != 0 si un écart dépasse la tolérance.
import { readFileSync } from "node:fs";
import { fileURLToPath } from "node:url";
import { dirname, join } from "node:path";
import { createRequire } from "node:module";

const __dirname = dirname(fileURLToPath(import.meta.url));
const require = createRequire(import.meta.url);
const { featureStack } = require(join(__dirname, "..", "web", "features.js"));
const { forwardProbs } = require(join(__dirname, "..", "web", "nn.js"));

const ref = JSON.parse(readFileSync(join(__dirname, "..", "artifacts", "verify_cases.json"), "utf-8"));
const W = {};
for (const k in ref.weights) {
  const bin = Buffer.from(ref.weights[k], "base64");
  W[k] = new Float32Array(bin.buffer, bin.byteOffset, bin.length / 4);
}
const tol = ref.tol ?? 2e-3;
let maxErr = 0, failed = 0;
for (const c of ref.cases) {
  const fs = featureStack(Float32Array.from(c.waveform));
  const { probs } = forwardProbs(fs.data, fs.nMels, fs.nFrames, W, ref.labels.length);
  let err = 0, best = 0;
  for (let i = 0; i < probs.length; i++) { err = Math.max(err, Math.abs(probs[i] - c.probs[i])); if (probs[i] > probs[best]) best = i; }
  maxErr = Math.max(maxErr, err);
  const okLabel = best === c.pred;
  if (err > tol || !okLabel) { console.error(`FAIL ${c.name}: label js=${best} py=${c.pred} err=${err.toExponential(2)}`); failed++; }
  else console.log(`ok   ${c.name}: label=${ref.labels[best]} err=${err.toExponential(2)}`);
}
console.log(`écart proba max = ${maxErr.toExponential(3)} (tol ${tol})`);
process.exit(failed ? 1 : 0);
