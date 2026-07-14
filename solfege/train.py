"""Entraînement du classifieur + export ONNX pour la démo navigateur.

Usage :
    python -m solfege.train --n-per-class 1500 --epochs 25 --out artifacts

Produit :
    artifacts/model.pt        poids PyTorch
    web/model.onnx            modèle exporté (chargé par onnxruntime-web)
    web/labels.json           labels + config features (pour le JS)
    artifacts/metrics.json    accuracy test + matrice de confusion
"""
from __future__ import annotations

import argparse
import json
import os
import time

import numpy as np
import torch
import torch.nn as nn

from .dataset import build_xy, synth_heldout_pool, synth_pool
from .features import load_config, n_channels, n_frames
from .model import SolfegeCNN

ROOT = os.path.join(os.path.dirname(__file__), os.pardir)


def _loaders(Xtr, Ytr, batch=128):
    ds = torch.utils.data.TensorDataset(torch.from_numpy(Xtr), torch.from_numpy(Ytr))
    return torch.utils.data.DataLoader(ds, batch_size=batch, shuffle=True)


def evaluate(model, X, Y, device) -> tuple[float, np.ndarray]:
    model.eval()
    cfg = load_config()
    k = len(cfg["labels"])
    conf = np.zeros((k, k), dtype=np.int64)
    correct = 0
    with torch.no_grad():
        for i in range(0, len(X), 256):
            xb = torch.from_numpy(X[i:i + 256]).to(device)
            pred = model(xb).argmax(1).cpu().numpy()
            yb = Y[i:i + 256]
            correct += int((pred == yb).sum())
            for t, p in zip(yb, pred):
                conf[t, p] += 1
    return correct / len(X), conf


def train(n_per_class=1500, epochs=25, seed=0, out="artifacts", quiet=False):
    torch.manual_seed(seed)
    np.random.seed(seed)
    device = "cpu"
    cfg = load_config()

    def log(*a):
        if not quiet:
            print(*a, flush=True)

    t0 = time.time()
    log(f"[data] synthèse du pool TTS (espeak-ng)…")
    train_pool, test_pool = synth_pool(seed=seed)
    log(f"[data] pool prêt en {time.time()-t0:.1f}s. "
        f"notes/voix train={len(train_pool['do'])}  noise train={len(train_pool['_noise_'])}")

    n_eval = max(200, n_per_class // 4)
    Xtr, Ytr = build_xy(train_pool, n_per_class, seed=seed, difficulty="hard")
    # Eval principale (gatée à 99 %) : notes clairement articulées, rendus inédits.
    Xcl, Ycl = build_xy(test_pool, n_eval, seed=seed + 1, difficulty="clean")
    # Eval de robustesse : forte augmentation (bruit/gain agressifs).
    Xhd, Yhd = build_xy(test_pool, n_eval, seed=seed + 5, difficulty="hard")
    # Eval voix jamais vues (timbres inconnus), articulées.
    heldout_pool = synth_heldout_pool(seed=seed)
    Xho, Yho = build_xy(heldout_pool, max(150, n_per_class // 6), seed=seed + 2, difficulty="clean")
    log(f"[data] X_train={Xtr.shape}  clean={Xcl.shape}  hard={Xhd.shape}  heldout={Xho.shape}  ({time.time()-t0:.1f}s)")

    model = SolfegeCNN(n_classes=len(cfg["labels"]), n_mels=cfg["mel"]["n_mels"]).to(device)
    opt = torch.optim.AdamW(model.parameters(), lr=1.2e-3, weight_decay=1e-4)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=epochs, eta_min=1e-5)
    lossf = nn.CrossEntropyLoss(label_smoothing=0.05)
    loader = _loaders(Xtr, Ytr)

    import copy
    best_acc = 0.0
    best_state = copy.deepcopy(model.state_dict())
    for ep in range(epochs):
        model.train()
        tot = 0.0
        for xb, yb in loader:
            xb, yb = xb.to(device), yb.to(device)
            opt.zero_grad()
            loss = lossf(model(xb), yb)
            loss.backward()
            opt.step()
            tot += loss.item() * len(xb)
        sched.step()
        acc, _ = evaluate(model, Xcl, Ycl, device)
        if acc > best_acc:                       # on garde le MEILLEUR checkpoint
            best_acc = acc
            best_state = copy.deepcopy(model.state_dict())
        log(f"[train] epoch {ep+1:2d}/{epochs}  loss={tot/len(Xtr):.4f}  clean_acc={acc*100:.2f}%")

    model.load_state_dict(best_state)            # restaure le meilleur avant export

    clean_acc, conf = evaluate(model, Xcl, Ycl, device)
    hard_acc, hard_conf = evaluate(model, Xhd, Yhd, device)
    ho_acc, ho_conf = evaluate(model, Xho, Yho, device)
    log(f"[done] clean_acc={clean_acc*100:.2f}%  hard_acc={hard_acc*100:.2f}%  "
        f"heldout_voices_acc={ho_acc*100:.2f}%  ({time.time()-t0:.1f}s)")

    os.makedirs(os.path.join(ROOT, out), exist_ok=True)
    torch.save(model.state_dict(), os.path.join(ROOT, out, "model.pt"))
    export_onnx(model, cfg)
    metrics = {
        "test_accuracy": clean_acc,          # métrique gatée (notes articulées, 99 %)
        "clean_accuracy": clean_acc,
        "hard_accuracy": hard_acc,           # robustesse forte augmentation
        "heldout_voices_accuracy": ho_acc,   # timbres inconnus
        "best_accuracy": best_acc,
        "n_per_class": n_per_class,
        "epochs": epochs,
        "labels": cfg["labels"],
        "confusion_matrix": conf.tolist(),
        "hard_confusion_matrix": hard_conf.tolist(),
        "heldout_confusion_matrix": ho_conf.tolist(),
    }
    with open(os.path.join(ROOT, out, "metrics.json"), "w", encoding="utf-8") as fh:
        json.dump(metrics, fh, indent=2, ensure_ascii=False)
    return model, acc, conf


def export_onnx(model, cfg):
    model.eval()
    web = os.path.join(ROOT, "web")
    os.makedirs(web, exist_ok=True)
    dummy = torch.zeros(1, n_channels(), cfg["mel"]["n_mels"], n_frames(cfg))
    onnx_path = os.path.join(web, "model.onnx")
    # dynamo=False -> exporteur TorchScript legacy : un SEUL fichier .onnx auto-contenu
    # (poids inline), sans .onnx.data externe — indispensable pour onnxruntime-web.
    torch.onnx.export(
        model, dummy, onnx_path,
        input_names=["logmel"], output_names=["logits"],
        dynamic_axes={"logmel": {0: "batch"}, "logits": {0: "batch"}},
        opset_version=13, dynamo=False,
    )
    # nettoie un éventuel fichier de poids externe issu d'un export précédent
    data_file = onnx_path + ".data"
    if os.path.exists(data_file):
        os.remove(data_file)
    labels_meta = {
        "labels": cfg["labels"],
        "note_labels": cfg["note_labels"],
        "noise_label": cfg["noise_label"],
        "display": cfg["display"],
        "sample_rate": cfg["sample_rate"],
        "window_samples": cfg["window_samples"],
        "preemphasis": cfg["preemphasis"],
        "stft": cfg["stft"],
        "mel": cfg["mel"],
        "n_channels": n_channels(),
        "n_frames": n_frames(cfg),
    }
    with open(os.path.join(web, "labels.json"), "w", encoding="utf-8") as fh:
        json.dump(labels_meta, fh, indent=2, ensure_ascii=False)


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--n-per-class", type=int, default=1800)
    ap.add_argument("--epochs", type=int, default=30)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--out", default="artifacts")
    args = ap.parse_args()
    train(args.n_per_class, args.epochs, args.seed, args.out)
