// Démo navigateur : micro -> log-mel (features.js) -> CNN (model.onnx via onnxruntime-web)
// -> décodage en flux (fusion des répétitions, suppression du bruit) -> suite de notes.

let session = null;
let meta = null;
let audioCtx = null;
let stream = null;
let processor = null;
let running = false;

const ring = { buf: null, size: 0, filled: 0 }; // buffer circulaire ~1 s
let sinceInfer = 0;
let lastEmitted = null;
let stableCount = 0;
const THRESHOLD = 0.6;
const MIN_STABLE = 1; // fenêtres consécutives requises pour émettre une note

const $ = (id) => document.getElementById(id);

async function loadModel() {
  meta = await (await fetch("labels.json")).json();
  ring.size = meta.window_samples;
  ring.buf = new Float32Array(ring.size);
  try {
    session = await ort.InferenceSession.create("model.onnx", {
      executionProviders: ["wasm"],
    });
    setStatus("modèle chargé ✔ — prêt");
    renderBars(meta.labels.map(() => 0));
  } catch (e) {
    setStatus("échec du chargement du modèle : " + e.message);
    console.error(e);
  }
  fetch("metrics.json").then(r => r.ok ? r.json() : null).then(m => {
    if (m) $("metrics").textContent =
      `Précision test (rendus TTS inédits) : ${(m.test_accuracy * 100).toFixed(2)} %` +
      (m.heldout_voices_accuracy != null
        ? ` · voix jamais vues : ${(m.heldout_voices_accuracy * 100).toFixed(1)} %` : "");
  }).catch(() => {});
  buildSampleButtons();
}

function setStatus(t) { $("status").textContent = t; }

// --- inférence sur une fenêtre (Float32Array de window_samples) ---
async function infer(frame) {
  const { data, nChannels, nMels, nFrames } = featureStack(frame);
  const tensor = new ort.Tensor("float32", data, [1, nChannels, nMels, nFrames]);
  const out = await session.run({ logmel: tensor });
  const logits = out.logits.data;
  // softmax
  let mx = -Infinity;
  for (const v of logits) mx = Math.max(mx, v);
  let sum = 0;
  const probs = new Float32Array(logits.length);
  for (let i = 0; i < logits.length; i++) { probs[i] = Math.exp(logits[i] - mx); sum += probs[i]; }
  for (let i = 0; i < probs.length; i++) probs[i] /= sum;
  let best = 0;
  for (let i = 1; i < probs.length; i++) if (probs[i] > probs[best]) best = i;
  return { label: meta.labels[best], conf: probs[best], probs };
}

// --- décodage en flux : met à jour l'affichage et la transcription ---
function onPrediction(label, conf, probs) {
  renderBars(probs);
  const isNote = label !== meta.noise_label && conf >= THRESHOLD;
  $("current").textContent = isNote ? meta.display[label] : "—";
  $("current").className = "current" + (isNote ? " active" : "");

  const token = isNote ? label : meta.noise_label;
  if (token === lastEmitted) { stableCount++; return; }
  // nouvelle valeur stable
  if (isNote) {
    stableCount = 1;
    if (stableCount >= MIN_STABLE) appendNote(label);
  }
  lastEmitted = token;
}

function appendNote(label) {
  const span = document.createElement("span");
  span.className = "note";
  span.textContent = meta.display[label];
  $("transcript").appendChild(span);
  $("transcript").scrollLeft = $("transcript").scrollWidth;
}

function renderBars(probs) {
  const bars = $("bars");
  if (bars.children.length !== meta.labels.length) {
    bars.innerHTML = "";
    meta.labels.forEach((l) => {
      const b = document.createElement("div");
      b.className = "bar";
      b.innerHTML = `<div class="fill"></div><span>${meta.display[l] === "—" ? "bruit" : meta.display[l]}</span>`;
      bars.appendChild(b);
    });
  }
  meta.labels.forEach((l, i) => {
    const fill = bars.children[i].querySelector(".fill");
    fill.style.height = Math.round(probs[i] * 100) + "%";
    fill.classList.toggle("hot", probs[i] === Math.max(...probs));
  });
}

// --- capture micro ---
async function startMic() {
  if (!session) { setStatus("modèle non chargé"); return; }
  stream = await navigator.mediaDevices.getUserMedia({ audio: { channelCount: 1 } });
  audioCtx = new (window.AudioContext || window.webkitAudioContext)({ sampleRate: meta.sample_rate });
  const src = audioCtx.createMediaStreamSource(stream);
  processor = audioCtx.createScriptProcessor(2048, 1, 1);
  ring.buf.fill(0); ring.filled = 0; sinceInfer = 0;
  lastEmitted = null; stableCount = 0;
  const hopSamples = Math.round(0.25 * meta.sample_rate);

  processor.onaudioprocess = (e) => {
    const input = e.inputBuffer.getChannelData(0);
    // pousse dans le buffer circulaire (décalage)
    const n = input.length;
    ring.buf.copyWithin(0, n);
    ring.buf.set(input, ring.size - n);
    ring.filled = Math.min(ring.size, ring.filled + n);
    sinceInfer += n;
    if (sinceInfer >= hopSamples && ring.filled >= ring.size) {
      sinceInfer = 0;
      const frame = ring.buf.slice();
      infer(frame).then(({ label, conf, probs }) => onPrediction(label, conf, probs));
    }
  };
  src.connect(processor);
  processor.connect(audioCtx.destination);
  running = true;
  $("mic").disabled = true; $("stop").disabled = false;
  setStatus("écoute en cours… chantez do ré mi fa sol la si");
}

function stopMic() {
  running = false;
  if (processor) processor.disconnect();
  if (stream) stream.getTracks().forEach((t) => t.stop());
  if (audioCtx) audioCtx.close();
  $("mic").disabled = false; $("stop").disabled = true;
  setStatus("arrêté");
}

// --- lecture d'un fichier wav (échantillons de test) ---
async function classifyWav(url, btn) {
  const arr = await (await fetch(url)).arrayBuffer();
  const ctx = new (window.AudioContext || window.webkitAudioContext)();
  const audio = await ctx.decodeAudioData(arr.slice(0));
  // rééchantillonne vers meta.sample_rate si besoin
  let data = audio.getChannelData(0);
  if (audio.sampleRate !== meta.sample_rate) {
    const off = new OfflineAudioContext(1, Math.ceil(audio.duration * meta.sample_rate), meta.sample_rate);
    const s = off.createBufferSource();
    s.buffer = audio; s.connect(off.destination); s.start();
    data = (await off.startRendering()).getChannelData(0);
  }
  new Audio(url).play().catch(() => {});
  // décodage en flux du clip
  const win = meta.window_samples, hop = Math.round(0.25 * meta.sample_rate);
  const seq = [];
  let prev = null;
  for (let start = 0; start + win <= data.length || start === 0; start += hop) {
    const frame = new Float32Array(win);
    frame.set(data.subarray(start, Math.min(start + win, data.length)));
    const { label, conf } = await infer(frame);
    const tok = conf >= THRESHOLD ? label : meta.noise_label;
    if (tok !== prev && tok !== meta.noise_label) seq.push(meta.display[tok]);
    prev = tok;
    if (start + win >= data.length) break;
  }
  btn.querySelector(".result").textContent = seq.length ? "→ " + seq.join(" ") : "→ (bruit)";
  ctx.close();
}

function buildSampleButtons() {
  fetch("samples/index.json").then(r => r.ok ? r.json() : []).then((list) => {
    const box = $("samples");
    box.innerHTML = "";
    (list || []).forEach((s) => {
      const btn = document.createElement("button");
      btn.className = "sample";
      btn.innerHTML = `<span class="name">▶ ${s.label}</span><span class="result"></span>`;
      btn.onclick = () => { if (session) classifyWav("samples/" + s.file, btn); };
      box.appendChild(btn);
    });
    if (!list || !list.length) box.innerHTML = '<p class="hint">Aucun échantillon (généré au build).</p>';
  }).catch(() => {});
}

$("mic").onclick = startMic;
$("stop").onclick = stopMic;
loadModel();
