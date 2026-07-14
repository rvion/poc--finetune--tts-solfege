"""Petit classifieur convolutionnel de mots-clés (keyword spotting).

Entrée : log-mel ``(batch, 1, n_mels, n_frames)``.
Sortie : logits sur les 8 classes (7 notes + _noise_).

~150k paramètres : entraînement rapide sur CPU, export ONNX propre, inférence
temps réel dans le navigateur via onnxruntime-web.
"""
from __future__ import annotations

import torch
import torch.nn as nn


class SolfegeCNN(nn.Module):
    """CNN de keyword-spotting.

    Point clé : le *pooling global* combine **moyenne + maximum** sur tout le plan
    temps-fréquence. Le max préserve la trace d'un onset consonantique bref
    (le « d » de *do*, le « s » de *sol*) qu'une simple moyenne diluerait — c'est
    lui qui sépare des voyelles tenues identiques (do/sol -> « ooo », mi/si -> « iii »).
    """

    def __init__(self, n_classes: int = 8, n_mels: int = 40, in_ch: int = 2):
        super().__init__()
        self.features = nn.Sequential(
            nn.Conv2d(in_ch, 32, 3, padding=1), nn.BatchNorm2d(32), nn.ReLU(),
            nn.MaxPool2d(2),
            nn.Conv2d(32, 64, 3, padding=1), nn.BatchNorm2d(64), nn.ReLU(),
            nn.MaxPool2d(2),
            nn.Conv2d(64, 96, 3, padding=1), nn.BatchNorm2d(96), nn.ReLU(),
            nn.MaxPool2d(2),
            nn.Conv2d(96, 96, 3, padding=1), nn.BatchNorm2d(96), nn.ReLU(),
        )
        self.avg = nn.AdaptiveAvgPool2d(1)
        self.mx = nn.AdaptiveMaxPool2d(1)
        self.head = nn.Sequential(
            nn.Flatten(),
            nn.Dropout(0.3),
            nn.Linear(96 * 2, 128), nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(128, n_classes),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        f = self.features(x)
        pooled = torch.cat([self.avg(f), self.mx(f)], dim=1)
        return self.head(pooled)
