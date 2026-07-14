"""Extraction de features log-mel — implémentation numpy *pure*.

Contrainte de conception : chaque étape doit être reproductible **à l'identique**
en JavaScript (voir ``web/features.js``) pour que le modèle entraîné en Python
donne exactement les mêmes entrées dans le navigateur. On n'utilise donc PAS
``librosa`` ni ``scipy.signal.stft`` (dont les conventions de fenêtrage/normalisation
sont difficiles à répliquer) : on réimplémente STFT + banc de filtres mel à la main.

La parité Python<->JS est vérifiée par ``tests/test_features.py`` contre les
vecteurs de référence ``tests/fixtures/feature_parity.json``.
"""
from __future__ import annotations

import json
import os
from functools import lru_cache

import numpy as np

CONFIG_PATH = os.path.join(os.path.dirname(__file__), os.pardir, "config.json")


@lru_cache(maxsize=1)
def load_config() -> dict:
    with open(CONFIG_PATH, "r", encoding="utf-8") as fh:
        return json.load(fh)


def hann_window(n: int) -> np.ndarray:
    """Fenêtre de Hann *périodique* (comme numpy.hanning(n+1)[:-1] / torch.hann_window).

    Formule explicite pour être copiable tel quel en JS.
    """
    k = np.arange(n, dtype=np.float64)
    return 0.5 - 0.5 * np.cos(2.0 * np.pi * k / n)


def hz_to_mel(f: np.ndarray | float) -> np.ndarray | float:
    """Échelle mel HTK (formule simple, facile à répliquer en JS)."""
    return 2595.0 * np.log10(1.0 + np.asarray(f, dtype=np.float64) / 700.0)


def mel_to_hz(m: np.ndarray | float) -> np.ndarray | float:
    return 700.0 * (np.power(10.0, np.asarray(m, dtype=np.float64) / 2595.0) - 1.0)


@lru_cache(maxsize=4)
def mel_filterbank(sample_rate: int, n_fft: int, n_mels: int, fmin: float, fmax: float) -> np.ndarray:
    """Banc de filtres triangulaires mel HTK, non normalisé (pics à 1.0).

    Retourne une matrice ``(n_mels, n_fft//2 + 1)``.
    """
    n_bins = n_fft // 2 + 1
    fft_freqs = np.linspace(0.0, sample_rate / 2.0, n_bins, dtype=np.float64)

    mel_min = hz_to_mel(fmin)
    mel_max = hz_to_mel(fmax)
    mel_points = np.linspace(mel_min, mel_max, n_mels + 2)
    hz_points = mel_to_hz(mel_points)

    fb = np.zeros((n_mels, n_bins), dtype=np.float64)
    for i in range(n_mels):
        left, center, right = hz_points[i], hz_points[i + 1], hz_points[i + 2]
        # branche montante
        rising = (fft_freqs - left) / max(center - left, 1e-9)
        # branche descendante
        falling = (right - fft_freqs) / max(right - center, 1e-9)
        fb[i] = np.clip(np.minimum(rising, falling), 0.0, None)
    return fb


def _frame_signal(y: np.ndarray, win_length: int, hop_length: int) -> np.ndarray:
    """Découpe en trames sans centrage (``center=False``). Trames incomplètes ignorées."""
    if len(y) < win_length:
        y = np.pad(y, (0, win_length - len(y)))
    n_frames = 1 + (len(y) - win_length) // hop_length
    idx = np.arange(win_length)[None, :] + hop_length * np.arange(n_frames)[:, None]
    return y[idx]


def logmel(waveform: np.ndarray, cfg: dict | None = None) -> np.ndarray:
    """Waveform float32 mono [-1, 1] -> log-mel normalisé, forme ``(n_mels, n_frames)``.

    Étapes (identiques en JS) :
      1. pré-accentuation ``y[n] = x[n] - a * x[n-1]``
      2. trames de ``win_length`` au pas ``hop_length`` (center=False)
      3. fenêtre de Hann périodique
      4. rFFT sur ``n_fft`` (zero-pad), spectre de puissance ``|X|^2``
      5. banc de filtres mel HTK
      6. ``log(mel + 1e-6)``
      7. normalisation par énoncé (moyenne/écart-type globaux) -> robuste au gain
    """
    cfg = cfg or load_config()
    sr = cfg["sample_rate"]
    n_fft = cfg["stft"]["n_fft"]
    win = cfg["stft"]["win_length"]
    hop = cfg["stft"]["hop_length"]
    n_mels = cfg["mel"]["n_mels"]
    fmin = cfg["mel"]["fmin"]
    fmax = cfg["mel"]["fmax"]
    pre = cfg.get("preemphasis", 0.0)

    y = np.asarray(waveform, dtype=np.float64).reshape(-1)
    target = cfg["window_samples"]
    if len(y) < target:
        y = np.pad(y, (0, target - len(y)))
    else:
        y = y[:target]

    if pre:
        y = np.concatenate([[y[0]], y[1:] - pre * y[:-1]])

    frames = _frame_signal(y, win, hop)
    frames = frames * hann_window(win)[None, :]

    spectrum = np.fft.rfft(frames, n=n_fft, axis=1)
    power = (spectrum.real ** 2 + spectrum.imag ** 2)  # (n_frames, n_bins)

    fb = mel_filterbank(sr, n_fft, n_mels, fmin, fmax)  # (n_mels, n_bins)
    mel = power @ fb.T  # (n_frames, n_mels)
    logm = np.log(mel + 1e-6).T  # (n_mels, n_frames)

    # Normalisation par énoncé : soustraction de la moyenne globale (cepstral mean
    # normalization). En domaine log, un gain multiplicatif devient une constante
    # additive uniforme -> la soustraction de moyenne rend le tout invariant au gain.
    # Pas de division par l'écart-type (instable sur le silence). BatchNorm du modèle
    # gère la mise à l'échelle. Trivial à répliquer en JS.
    logm = logm - logm.mean()
    return logm.astype(np.float32)


def delta(feat: np.ndarray) -> np.ndarray:
    """Dérivée temporelle Δ : ``d[:,t] = (x[:,t+1] - x[:,t-1]) / 2`` (bords répliqués).

    Donne au modèle l'information de *transition* — l'attaque consonantique brève qui
    distingue fa/la et mi/si. Opération linéaire, répliquée à l'identique en JS.
    """
    padded = np.concatenate([feat[:, :1], feat, feat[:, -1:]], axis=1)
    return ((padded[:, 2:] - padded[:, :-2]) * 0.5).astype(np.float32)


def feature_stack(waveform: np.ndarray, cfg: dict | None = None) -> np.ndarray:
    """Entrée du modèle : 2 canaux ``[log-mel, Δ log-mel]``, forme ``(2, n_mels, n_frames)``."""
    logm = logmel(waveform, cfg)
    return np.stack([logm, delta(logm)], axis=0)


def n_channels() -> int:
    return 2


def n_frames(cfg: dict | None = None) -> int:
    cfg = cfg or load_config()
    win = cfg["stft"]["win_length"]
    hop = cfg["stft"]["hop_length"]
    return 1 + (cfg["window_samples"] - win) // hop
