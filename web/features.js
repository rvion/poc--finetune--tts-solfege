// Extraction log-mel — MIROIR EXACT de solfege/features.py.
// Toute divergence casse le modèle. La parité est vérifiée par tests/test_features.py
// contre tests/fixtures/feature_parity.json (mêmes vecteurs, tolérance 1e-4).
//
// Aucune dépendance externe : FFT maison (radix-2 itérative). n_fft = 512 (puissance de 2).

const FEAT = {
  sampleRate: 16000,
  windowSamples: 16000,
  preemphasis: 0.97,
  nFft: 512,
  winLength: 400,
  hopLength: 160,
  nMels: 40,
  fmin: 40.0,
  fmax: 8000.0,
};

function hzToMel(f) { return 2595.0 * Math.log10(1.0 + f / 700.0); }
function melToHz(m) { return 700.0 * (Math.pow(10.0, m / 2595.0) - 1.0); }

function hannWindow(n) {
  const w = new Float64Array(n);
  for (let k = 0; k < n; k++) w[k] = 0.5 - 0.5 * Math.cos((2.0 * Math.PI * k) / n);
  return w;
}

// Banc de filtres mel HTK triangulaire, non normalisé (identique à features.py).
function melFilterbank(sr, nFft, nMels, fmin, fmax) {
  const nBins = nFft / 2 + 1;
  const fftFreqs = new Float64Array(nBins);
  for (let i = 0; i < nBins; i++) fftFreqs[i] = (i * (sr / 2.0)) / (nBins - 1);
  const melMin = hzToMel(fmin), melMax = hzToMel(fmax);
  const hz = new Float64Array(nMels + 2);
  for (let i = 0; i < nMels + 2; i++) {
    hz[i] = melToHz(melMin + ((melMax - melMin) * i) / (nMels + 1));
  }
  const fb = [];
  for (let m = 0; m < nMels; m++) {
    const row = new Float64Array(nBins);
    const left = hz[m], center = hz[m + 1], right = hz[m + 2];
    for (let k = 0; k < nBins; k++) {
      const rising = (fftFreqs[k] - left) / Math.max(center - left, 1e-9);
      const falling = (right - fftFreqs[k]) / Math.max(right - center, 1e-9);
      row[k] = Math.max(0.0, Math.min(rising, falling));
    }
    fb.push(row);
  }
  return fb;
}

// FFT réelle via FFT complexe radix-2 itérative (bit-reversal + papillons).
function fftComplex(re, im) {
  const n = re.length;
  for (let i = 1, j = 0; i < n; i++) {
    let bit = n >> 1;
    for (; j & bit; bit >>= 1) j ^= bit;
    j ^= bit;
    if (i < j) { [re[i], re[j]] = [re[j], re[i]]; [im[i], im[j]] = [im[j], im[i]]; }
  }
  for (let len = 2; len <= n; len <<= 1) {
    const ang = (-2.0 * Math.PI) / len;
    const wr = Math.cos(ang), wi = Math.sin(ang);
    for (let i = 0; i < n; i += len) {
      let cwr = 1.0, cwi = 0.0;
      for (let k = 0; k < len / 2; k++) {
        const ur = re[i + k], ui = im[i + k];
        const vr = re[i + k + len / 2] * cwr - im[i + k + len / 2] * cwi;
        const vi = re[i + k + len / 2] * cwi + im[i + k + len / 2] * cwr;
        re[i + k] = ur + vr; im[i + k] = ui + vi;
        re[i + k + len / 2] = ur - vr; im[i + k + len / 2] = ui - vi;
        const nwr = cwr * wr - cwi * wi;
        cwi = cwr * wi + cwi * wr; cwr = nwr;
      }
    }
  }
}

const _fb = melFilterbank(FEAT.sampleRate, FEAT.nFft, FEAT.nMels, FEAT.fmin, FEAT.fmax);
const _hann = hannWindow(FEAT.winLength);

// waveform Float32/Float64 [-1,1] -> Float32Array de longueur nMels*nFrames (row-major mel x temps).
function logmel(waveform) {
  const total = FEAT.windowSamples;
  let y = new Float64Array(total);
  const L = Math.min(waveform.length, total);
  for (let i = 0; i < L; i++) y[i] = waveform[i];

  // pré-accentuation
  if (FEAT.preemphasis) {
    const pe = new Float64Array(total);
    pe[0] = y[0];
    for (let i = 1; i < total; i++) pe[i] = y[i] - FEAT.preemphasis * y[i - 1];
    y = pe;
  }

  const win = FEAT.winLength, hop = FEAT.hopLength, nFft = FEAT.nFft;
  const nBins = nFft / 2 + 1;
  const nFrames = 1 + Math.floor((total - win) / hop);
  const mel = new Float64Array(FEAT.nMels * nFrames);

  const re = new Float64Array(nFft), im = new Float64Array(nFft);
  for (let t = 0; t < nFrames; t++) {
    const off = t * hop;
    re.fill(0); im.fill(0);
    for (let k = 0; k < win; k++) re[k] = y[off + k] * _hann[k];
    fftComplex(re, im);
    // spectre de puissance
    const power = new Float64Array(nBins);
    for (let b = 0; b < nBins; b++) power[b] = re[b] * re[b] + im[b] * im[b];
    for (let m = 0; m < FEAT.nMels; m++) {
      const row = _fb[m];
      let s = 0.0;
      for (let b = 0; b < nBins; b++) s += power[b] * row[b];
      mel[m * nFrames + t] = Math.log(s + 1e-6);
    }
  }

  // normalisation par énoncé : soustraction de la moyenne globale
  let mean = 0.0;
  for (let i = 0; i < mel.length; i++) mean += mel[i];
  mean /= mel.length;
  const out = new Float32Array(mel.length);
  for (let i = 0; i < mel.length; i++) out[i] = mel[i] - mean;
  return { data: out, nMels: FEAT.nMels, nFrames };
}

// Dérivée temporelle Δ (miroir de features.delta) : d[:,t]=(x[:,t+1]-x[:,t-1])/2,
// bords répliqués. Entrée/sortie en layout row-major (mel x temps), longueur nMels*nFrames.
function deltaChannel(logm, nMels, nFrames) {
  const d = new Float32Array(nMels * nFrames);
  for (let m = 0; m < nMels; m++) {
    const base = m * nFrames;
    for (let t = 0; t < nFrames; t++) {
      const prev = logm[base + Math.max(0, t - 1)];
      const next = logm[base + Math.min(nFrames - 1, t + 1)];
      d[base + t] = (next - prev) * 0.5;
    }
  }
  return d;
}

// Entrée du modèle : 2 canaux [log-mel, Δ] concaténés, forme [1, 2, nMels, nFrames].
function featureStack(waveform) {
  const { data, nMels, nFrames } = logmel(waveform);
  const d = deltaChannel(data, nMels, nFrames);
  const out = new Float32Array(2 * nMels * nFrames);
  out.set(data, 0);
  out.set(d, nMels * nFrames);
  return { data: out, nChannels: 2, nMels, nFrames };
}

if (typeof module !== "undefined" && module.exports) {
  module.exports = { logmel, featureStack, deltaChannel, FEAT, hzToMel, melToHz, melFilterbank, fftComplex };
}
