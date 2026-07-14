"""Inférence : classification d'une fenêtre + décodage en flux (streaming).

Le décodeur en flux reproduit le comportement de la démo navigateur :
fenêtre glissante -> une prédiction par pas -> on fusionne les répétitions
consécutives et on **jette la classe _noise_** pour ne garder que la suite de notes.
C'est ce qui transcrit « ddooo…rémi euh… la » en ``["do","re","mi","la"]``.
"""
from __future__ import annotations

import os

import numpy as np

from .features import feature_stack, load_config

ROOT = os.path.join(os.path.dirname(__file__), os.pardir)


def softmax(x: np.ndarray, axis=-1) -> np.ndarray:
    x = x - x.max(axis=axis, keepdims=True)
    e = np.exp(x)
    return e / e.sum(axis=axis, keepdims=True)


class OnnxClassifier:
    """Charge ``web/model.onnx`` via onnxruntime (mêmes poids que la démo)."""

    def __init__(self, onnx_path: str | None = None):
        import onnxruntime as ort
        self.cfg = load_config()
        self.labels = self.cfg["labels"]
        path = onnx_path or os.path.join(ROOT, "web", "model.onnx")
        self.sess = ort.InferenceSession(path, providers=["CPUExecutionProvider"])
        self.inp = self.sess.get_inputs()[0].name

    def logits(self, feats: np.ndarray) -> np.ndarray:
        """feats: (2, n_mels, n_frames) ou batch (N, 2, n_mels, n_frames) -> logits."""
        if feats.ndim == 3:
            feats = feats[None]
        return self.sess.run(None, {self.inp: feats.astype(np.float32)})[0]

    def classify(self, wav: np.ndarray) -> tuple[str, float, np.ndarray]:
        probs = softmax(self.logits(feature_stack(wav, self.cfg))[0])
        i = int(probs.argmax())
        return self.labels[i], float(probs[i]), probs


class TorchClassifier:
    """Charge ``artifacts/model.pt`` (utile si onnxruntime absent)."""

    def __init__(self, pt_path: str | None = None):
        import torch
        from .model import SolfegeCNN
        self.cfg = load_config()
        self.labels = self.cfg["labels"]
        self.model = SolfegeCNN(len(self.labels), self.cfg["mel"]["n_mels"])
        path = pt_path or os.path.join(ROOT, "artifacts", "model.pt")
        self.model.load_state_dict(torch.load(path, map_location="cpu"))
        self.model.eval()
        self._torch = torch

    def classify(self, wav: np.ndarray) -> tuple[str, float, np.ndarray]:
        feats = feature_stack(wav, self.cfg)[None]
        with self._torch.no_grad():
            logits = self.model(self._torch.from_numpy(feats)).numpy()[0]
        probs = softmax(logits)
        i = int(probs.argmax())
        return self.labels[i], float(probs[i]), probs


def stream_decode(clf, wav: np.ndarray, hop_sec: float = 0.25,
                  threshold: float = 0.6, min_repeat: int = 1) -> list[str]:
    """Fenêtre glissante -> suite de notes (sans _noise_, répétitions fusionnées).

    ``threshold`` : confiance minimale pour retenir une note.
    ``min_repeat`` : nb de fenêtres consécutives requises pour valider une note
                     (anti-faux-positif ; 1 = désactivé).
    """
    cfg = clf.cfg
    sr = cfg["sample_rate"]
    win = cfg["window_samples"]
    hop = max(1, int(hop_sec * sr))
    noise = cfg["noise_label"]

    preds: list[str] = []
    for start in range(0, max(1, len(wav) - win + 1), hop):
        chunk = wav[start:start + win]
        if len(chunk) < win:
            chunk = np.pad(chunk, (0, win - len(chunk)))
        label, conf, _ = clf.classify(chunk)
        preds.append(label if conf >= threshold else noise)

    # fusion des répétitions consécutives + suppression du bruit
    out: list[str] = []
    prev = None
    run = 0
    for p in preds + [None]:
        if p == prev:
            run += 1
            continue
        if prev is not None and prev != noise and run >= min_repeat:
            out.append(prev)
        prev, run = p, 1
    return out
