"""Génère une page HTML **auto-contenue** (modèle embarqué, inférence JS pure).

Aucune dépendance externe (ni CDN, ni fichier séparé) : les poids du CNN sont
encodés en base64 (Float32 little-endian) et injectés dans le template
``scripts/standalone_template.html``. La page tourne entièrement dans le navigateur
et peut être hébergée n'importe où (y compris un Artifact).

Usage : python scripts/export_standalone.py [--pt artifacts/model.pt] [--out web/standalone.html]
"""
from __future__ import annotations

import argparse
import base64
import json
import os

import numpy as np
import torch

from solfege.features import load_config, n_channels, n_frames
from solfege.model import SolfegeCNN

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
TEMPLATE = os.path.join(ROOT, "scripts", "standalone_template.html")


def b64(arr: np.ndarray) -> str:
    return base64.b64encode(arr.astype("<f4").tobytes()).decode("ascii")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--pt", default=os.path.join(ROOT, "artifacts", "model.pt"))
    ap.add_argument("--out", default=os.path.join(ROOT, "web", "standalone.html"))
    ap.add_argument("--body-only", default=None,
                    help="écrit aussi une version contenu-seul (pour Artifact) à ce chemin")
    args = ap.parse_args()

    cfg = load_config()
    model = SolfegeCNN(len(cfg["labels"]), cfg["mel"]["n_mels"], in_ch=n_channels())
    model.load_state_dict(torch.load(args.pt, map_location="cpu"))
    model.eval()

    weights = {k: b64(v.detach().cpu().numpy().reshape(-1)) for k, v in model.state_dict().items()
               if v.dtype.is_floating_point}

    meta = {
        "labels": cfg["labels"],
        "note_labels": cfg["note_labels"],
        "noise_label": cfg["noise_label"],
        "display": cfg["display"],
        "features": {
            "sample_rate": cfg["sample_rate"],
            "window_samples": cfg["window_samples"],
            "preemphasis": cfg["preemphasis"],
            "stft": cfg["stft"],
            "mel": cfg["mel"],
            "n_frames": n_frames(cfg),
        },
    }
    metrics_path = os.path.join(ROOT, "artifacts", "metrics.json")
    metrics = json.load(open(metrics_path)) if os.path.exists(metrics_path) else {
        "clean_accuracy": 0, "heldout_voices_accuracy": 0, "hard_accuracy": 0}

    features_js = open(os.path.join(ROOT, "web", "features.js"), encoding="utf-8").read()
    nn_js = open(os.path.join(ROOT, "web", "nn.js"), encoding="utf-8").read()

    body = open(TEMPLATE, encoding="utf-8").read()
    body = body.replace("__FEATURES_JS__", features_js)
    body = body.replace("__NN_JS__", nn_js)
    body = body.replace("__META_JSON__", json.dumps(meta, ensure_ascii=False))
    body = body.replace("__WEIGHTS_JSON__", json.dumps(weights))
    body = body.replace("__METRICS_JSON__", json.dumps({
        "clean_accuracy": metrics.get("clean_accuracy", 0),
        "heldout_voices_accuracy": metrics.get("heldout_voices_accuracy", 0),
        "hard_accuracy": metrics.get("hard_accuracy", 0),
    }))

    # version contenu-seul (pour Artifact : pas de <html>/<head>/<body>)
    if args.body_only:
        with open(args.body_only, "w", encoding="utf-8") as fh:
            fh.write(body)
        print(f"[standalone] contenu-seul -> {args.body_only} ({len(body)//1024} Ko)")

    # version page complète (pour le repo / hébergement direct)
    full = ("<!doctype html><html lang=\"fr\"><head><meta charset=\"utf-8\">"
            "<meta name=\"viewport\" content=\"width=device-width, initial-scale=1\">"
            "<title>Solfège STT — démo autonome</title></head><body>" + body + "</body></html>")
    with open(args.out, "w", encoding="utf-8") as fh:
        fh.write(full)
    print(f"[standalone] page complète -> {args.out} ({len(full)//1024} Ko)")


if __name__ == "__main__":
    main()
