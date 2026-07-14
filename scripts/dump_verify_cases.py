"""Produit artifacts/verify_cases.json : waveforms de test + probas PyTorch attendues
+ poids du modèle (base64), pour la vérification JS pur (scripts/verify_standalone.mjs)."""
from __future__ import annotations

import base64
import json
import os

import numpy as np
import torch

from solfege.augment import add_noise, time_stretch
from solfege.features import feature_stack, load_config, n_channels
from solfege.infer import softmax
from solfege.model import SolfegeCNN
from solfege.tts import NOTE_TEXT, TTSParams, synth, trim_silence

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


def b64(a):
    return base64.b64encode(a.astype("<f4").tobytes()).decode("ascii")


def main():
    cfg = load_config()
    model = SolfegeCNN(len(cfg["labels"]), cfg["mel"]["n_mels"], in_ch=n_channels())
    model.load_state_dict(torch.load(os.path.join(ROOT, "artifacts", "model.pt"), map_location="cpu"))
    model.eval()

    rng = np.random.default_rng(3)
    cases = []
    # notes claires, une note tenue, un parasite, du bruit
    specs = [("do", "fr-fr", 150, None), ("mi", "it", 200, None), ("sol", "es", 130, None),
             ("la", "fr-be", 170, 0.5), ("si", "fr-fr", 140, None),
             ("euh", "fr-fr", 160, None), ("bonjour", "fr-fr", 160, None)]
    for i, (word, voice, speed, stretch) in enumerate(specs):
        text = NOTE_TEXT.get(word, word)
        y = trim_silence(synth(text, TTSParams(voice, speed, 50)))
        if stretch:
            y = time_stretch(y, stretch)
        y = add_noise(y, 30, rng)
        y = np.clip(y, -1, 1).astype(np.float32)[: cfg["window_samples"]]
        feats = feature_stack(y, cfg)[None]
        with torch.no_grad():
            logits = model(torch.from_numpy(feats)).numpy()[0]
        probs = softmax(logits)
        cases.append({
            "name": f"{i}_{word}",
            "waveform": [round(float(v), 7) for v in y],
            "probs": [round(float(v), 6) for v in probs],
            "pred": int(probs.argmax()),
        })

    weights = {k: b64(v.detach().cpu().numpy().reshape(-1))
               for k, v in model.state_dict().items() if v.dtype.is_floating_point}
    out = {"tol": 2e-3, "labels": cfg["labels"], "cases": cases, "weights": weights}
    path = os.path.join(ROOT, "artifacts", "verify_cases.json")
    with open(path, "w", encoding="utf-8") as fh:
        json.dump(out, fh)
    print(f"[verify] {len(cases)} cas + poids -> {path}")


if __name__ == "__main__":
    main()
