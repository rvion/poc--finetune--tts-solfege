"""Génère des extraits .wav de démonstration (espeak-ng) pour la démo web.

Ces clips permettent de tester le modèle dans le navigateur SANS micro. Ils sont
écrits dans ``web/samples/`` avec un ``index.json`` listé par ``app.js``.

Usage : python scripts/gen_samples.py
"""
from __future__ import annotations

import json
import os

import numpy as np
from scipy.io import wavfile

from solfege.augment import add_noise, time_stretch
from solfege.features import load_config
from solfege.tts import TTSParams, synth, trim_silence

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
OUT = os.path.join(ROOT, "web", "samples")


def write_wav(name, y, sr):
    y = np.clip(y, -1, 1)
    wavfile.write(os.path.join(OUT, name), sr, (y * 32767).astype(np.int16))


def concat(clips, gap_sec, sr):
    gap = np.zeros(int(gap_sec * sr), dtype=np.float32)
    out = []
    for c in clips:
        out.append(c)
        out.append(gap)
    return np.concatenate(out) if out else np.zeros(sr, dtype=np.float32)


def main():
    os.makedirs(OUT, exist_ok=True)
    cfg = load_config()
    sr = cfg["sample_rate"]
    rng = np.random.default_rng(7)
    notes = [("do", "do"), ("re", "ré"), ("mi", "mi"), ("fa", "fa"),
             ("sol", "sol"), ("la", "la"), ("si", "si")]
    index = []

    # 1) gamme lente et claire
    scale = [trim_silence(synth(t, TTSParams("fr-fr", 150, 50), sr)) for _, t in notes]
    y = concat(scale, 0.8, sr)
    write_wav("scale_slow.wav", y, sr)
    index.append({"label": "Gamme lente : do ré mi fa sol la si", "file": "scale_slow.wav"})

    # 2) gamme rapide
    fast = [trim_silence(synth(t, TTSParams("it", 320, 55), sr)) for _, t in notes]
    y = concat(fast, 0.5, sr)
    write_wav("scale_fast.wav", y, sr)
    index.append({"label": "Gamme rapide", "file": "scale_fast.wav"})

    # 3) notes tenues « dooo rééé miii »
    held = [time_stretch(trim_silence(synth(t, TTSParams("fr-fr", 130, 48), sr)), 0.55)
            for _, t in notes[:4]]
    y = concat(held, 0.7, sr)
    write_wav("held_notes.wav", y, sr)
    index.append({"label": "Notes tenues : dooo rééé miii faaa", "file": "held_notes.wav"})

    # 4) bruit / mots parasites (doit être ignoré)
    junk = [trim_silence(synth(w, TTSParams("fr-fr", 160, 50), sr))
            for w in ["euh", "ah non", "attends", "voilà"]]
    y = add_noise(concat(junk, 0.2, sr), 25, rng)
    write_wav("garbage.wav", y, sr)
    index.append({"label": "Parasites : « euh… ah non… » (→ ignoré)", "file": "garbage.wav"})

    # 5) mélange réaliste : do ré euh mi ... sol
    mix = [
        trim_silence(synth("do", TTSParams("fr-fr", 150, 50), sr)),
        trim_silence(synth("ré", TTSParams("fr-fr", 150, 50), sr)),
        trim_silence(synth("euh", TTSParams("fr-fr", 150, 50), sr)),
        time_stretch(trim_silence(synth("mi", TTSParams("fr-fr", 140, 52), sr)), 0.6),
        trim_silence(synth("sol", TTSParams("fr-fr", 150, 50), sr)),
    ]
    y = add_noise(concat(mix, 0.7, sr), 28, rng)
    write_wav("mixed.wav", y, sr)
    index.append({"label": "Mélange : do ré (euh) mi sol", "file": "mixed.wav"})

    with open(os.path.join(OUT, "index.json"), "w", encoding="utf-8") as fh:
        json.dump(index, fh, indent=2, ensure_ascii=False)
    print(f"[samples] {len(index)} clips écrits dans {OUT}")


if __name__ == "__main__":
    main()
