// Forward du CNN (miroir EXACT de solfege/model.py) en JavaScript pur.
// Utilisé par la page autonome (web/standalone.html) et vérifié en node contre
// PyTorch (scripts/verify_standalone.mjs). Aucune dépendance externe.
//
// `weights` = dict des tenseurs du state_dict (Float32Array), clés PyTorch :
//   features.{0,4,8,12}.{weight,bias}  (Conv2d 3x3, pad 1, stride 1)
//   features.{1,5,9,13}.{weight,bias,running_mean,running_var}  (BatchNorm2d)
//   head.2.{weight,bias} (Linear 192->128), head.5.{weight,bias} (Linear 128->nClasses)

function conv2d(x, Ci, H, Wd, w, b, Co) {
  const K = 3, pad = 1, out = new Float32Array(Co * H * Wd);
  for (let oc = 0; oc < Co; oc++) {
    const wb = oc * Ci * K * K;
    for (let oy = 0; oy < H; oy++) for (let ox = 0; ox < Wd; ox++) {
      let s = b[oc];
      for (let ic = 0; ic < Ci; ic++) {
        const ib = ic * H * Wd, wcb = wb + ic * K * K;
        for (let ky = 0; ky < K; ky++) {
          const iy = oy + ky - pad; if (iy < 0 || iy >= H) continue;
          for (let kx = 0; kx < K; kx++) {
            const ix = ox + kx - pad; if (ix < 0 || ix >= Wd) continue;
            s += x[ib + iy * Wd + ix] * w[wcb + ky * K + kx];
          }
        }
      }
      out[oc * H * Wd + oy * Wd + ox] = s;
    }
  }
  return out;
}

function bnRelu(x, C, H, Wd, g, be, me, va) {
  const eps = 1e-5;
  for (let c = 0; c < C; c++) {
    const inv = 1 / Math.sqrt(va[c] + eps), gg = g[c] * inv, bb = be[c] - me[c] * gg;
    for (let i = c * H * Wd; i < (c + 1) * H * Wd; i++) { const v = x[i] * gg + bb; x[i] = v > 0 ? v : 0; }
  }
  return x;
}

function maxpool(x, C, H, Wd) {
  const Ho = H >> 1, Wo = Wd >> 1, out = new Float32Array(C * Ho * Wo);
  for (let c = 0; c < C; c++) for (let oy = 0; oy < Ho; oy++) for (let ox = 0; ox < Wo; ox++) {
    let m = -1e30;
    for (let dy = 0; dy < 2; dy++) for (let dx = 0; dx < 2; dx++) {
      const v = x[c * H * Wd + (oy * 2 + dy) * Wd + (ox * 2 + dx)]; if (v > m) m = v;
    }
    out[c * Ho * Wo + oy * Wo + ox] = m;
  }
  return { data: out, H: Ho, W: Wo };
}

function linear(x, inN, outN, w, b) {
  const out = new Float32Array(outN);
  for (let o = 0; o < outN; o++) { let s = b[o]; const wb = o * inN; for (let i = 0; i < inN; i++) s += w[wb + i] * x[i]; out[o] = s; }
  return out;
}

// featData : Float32Array (2 canaux [log-mel, Δ]) de longueur 2*H0*W0.
function forwardProbs(featData, H0, W0, W, nClasses) {
  let C = 2, H = H0, Wd = W0, x = featData;
  x = conv2d(x, C, H, Wd, W["features.0.weight"], W["features.0.bias"], 32); C = 32;
  x = bnRelu(x, C, H, Wd, W["features.1.weight"], W["features.1.bias"], W["features.1.running_mean"], W["features.1.running_var"]);
  ({ data: x, H, W: Wd } = maxpool(x, C, H, Wd));
  x = conv2d(x, C, H, Wd, W["features.4.weight"], W["features.4.bias"], 64); C = 64;
  x = bnRelu(x, C, H, Wd, W["features.5.weight"], W["features.5.bias"], W["features.5.running_mean"], W["features.5.running_var"]);
  ({ data: x, H, W: Wd } = maxpool(x, C, H, Wd));
  x = conv2d(x, C, H, Wd, W["features.8.weight"], W["features.8.bias"], 96); C = 96;
  x = bnRelu(x, C, H, Wd, W["features.9.weight"], W["features.9.bias"], W["features.9.running_mean"], W["features.9.running_var"]);
  ({ data: x, H, W: Wd } = maxpool(x, C, H, Wd));
  x = conv2d(x, C, H, Wd, W["features.12.weight"], W["features.12.bias"], 96); C = 96;
  x = bnRelu(x, C, H, Wd, W["features.13.weight"], W["features.13.bias"], W["features.13.running_mean"], W["features.13.running_var"]);
  // pooling global moyenne + max
  const avg = new Float32Array(C), mx = new Float32Array(C);
  for (let c = 0; c < C; c++) {
    let s = 0, m = -1e30;
    for (let i = c * H * Wd; i < (c + 1) * H * Wd; i++) { s += x[i]; if (x[i] > m) m = x[i]; }
    avg[c] = s / (H * Wd); mx[c] = m;
  }
  const pooled = new Float32Array(2 * C); pooled.set(avg, 0); pooled.set(mx, C);
  let h = linear(pooled, 2 * C, 128, W["head.2.weight"], W["head.2.bias"]);
  for (let i = 0; i < h.length; i++) if (h[i] < 0) h[i] = 0;
  const logits = linear(h, 128, nClasses, W["head.5.weight"], W["head.5.bias"]);
  let mxl = -1e30; for (const v of logits) if (v > mxl) mxl = v;
  let sum = 0; const p = new Float32Array(logits.length);
  for (let i = 0; i < p.length; i++) { p[i] = Math.exp(logits[i] - mxl); sum += p[i]; }
  for (let i = 0; i < p.length; i++) p[i] /= sum;
  return { probs: p, logits };
}

if (typeof module !== "undefined" && module.exports) {
  module.exports = { forwardProbs, conv2d, bnRelu, maxpool, linear };
}
