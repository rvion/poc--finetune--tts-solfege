"""Génère les vecteurs de référence pour la parité features Python<->JS.

Ecrit ``tests/fixtures/feature_parity.json`` : pour chaque forme d'onde d'entrée,
la sortie log-mel attendue. Le test Python vérifie qu'il reproduit ces valeurs ;
le test JS (node) vérifie que ``web/features.js`` produit les mêmes (tolérance 1e-4).

Usage : python scripts/make_fixtures.py
"""
from __future__ import annotations

import json
import os

import numpy as np

from solfege.features import feature_stack, load_config

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
OUT = os.path.join(ROOT, "tests", "fixtures", "feature_parity.json")


def make_cases():
    cfg = load_config()
    sr = cfg["sample_rate"]
    rng = np.random.default_rng(0)
    cases = []

    # 1) sinus pur 440 Hz
    t = np.arange(6000) / sr
    cases.append(("sine440", 0.6 * np.sin(2 * np.pi * 440 * t)))
    # 2) sweep
    f = np.linspace(200, 2000, 8000)
    cases.append(("sweep", 0.5 * np.sin(2 * np.pi * np.cumsum(f) / sr)))
    # 3) bruit blanc déterministe
    cases.append(("noise", 0.3 * rng.standard_normal(5000)))
    # 4) impulsion + silence
    imp = np.zeros(4000); imp[100] = 1.0; imp[2000] = -0.8
    cases.append(("impulses", imp))
    return cfg, cases


def main():
    cfg, cases = make_cases()
    os.makedirs(os.path.dirname(OUT), exist_ok=True)
    out = {"sample_rate": cfg["sample_rate"], "tol": 1e-4, "cases": []}
    for name, wav in cases:
        feat = feature_stack(wav.astype(np.float32), cfg)   # (2, n_mels, n_frames)
        out["cases"].append({
            "name": name,
            "waveform": [round(float(v), 7) for v in wav],
            "shape": list(feat.shape),
            "features": [round(float(v), 6) for v in feat.reshape(-1)],
        })
    with open(OUT, "w", encoding="utf-8") as fh:
        json.dump(out, fh)
    print(f"[fixtures] {len(out['cases'])} cas écrits dans {OUT}")


if __name__ == "__main__":
    main()
