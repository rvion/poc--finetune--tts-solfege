// Vérifie que web/features.js reproduit les vecteurs de référence Python.
// Sort en code 0 si tout est dans la tolérance, 1 sinon. Appelé par test_features.py.
import { readFileSync } from "node:fs";
import { fileURLToPath } from "node:url";
import { dirname, join } from "node:path";
import { createRequire } from "node:module";

const __dirname = dirname(fileURLToPath(import.meta.url));
const require = createRequire(import.meta.url);
const { featureStack } = require(join(__dirname, "..", "web", "features.js"));

const fx = JSON.parse(readFileSync(join(__dirname, "fixtures", "feature_parity.json"), "utf-8"));
const tol = fx.tol ?? 1e-4;

let maxErr = 0;
let failed = 0;
for (const c of fx.cases) {
  const { data, nChannels, nMels, nFrames } = featureStack(Float32Array.from(c.waveform));
  // shape Python = [2, nMels, nFrames]
  if (nChannels !== c.shape[0] || nMels !== c.shape[1] || nFrames !== c.shape[2]) {
    console.error(`FAIL ${c.name}: forme JS [${nChannels},${nMels},${nFrames}] != Python [${c.shape}]`);
    failed++; continue;
  }
  let err = 0;
  for (let i = 0; i < data.length; i++) err = Math.max(err, Math.abs(data[i] - c.features[i]));
  maxErr = Math.max(maxErr, err);
  if (err > tol) { console.error(`FAIL ${c.name}: écart max ${err.toExponential(3)} > tol ${tol}`); failed++; }
  else console.log(`ok   ${c.name}: écart max ${err.toExponential(3)}`);
}
console.log(`écart max global = ${maxErr.toExponential(3)} (tol ${tol})`);
process.exit(failed ? 1 : 0);
