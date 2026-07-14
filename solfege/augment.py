"""Augmentation de données audio.

Objectifs :
  * simuler les notes **tenues** (« dooooooo ») par étirement temporel (vocodeur de phase)
    qui allonge la voyelle sans changer la hauteur ;
  * simuler la variabilité réelle : bruit de fond, gain, position dans la fenêtre,
    léger décalage de hauteur.

Tout est piloté par un ``numpy.random.Generator`` explicite pour la reproductibilité.
"""
from __future__ import annotations

import numpy as np


def time_stretch(y: np.ndarray, rate: float, n_fft: int = 512, hop: int = 128) -> np.ndarray:
    """Étire dans le temps par un facteur ``1/rate`` (vocodeur de phase).

    ``rate < 1`` -> plus long (note tenue) ; ``rate > 1`` -> plus court (lecture rapide).
    La hauteur (pitch) est préservée.
    """
    if abs(rate - 1.0) < 1e-3 or len(y) < n_fft:
        return y.astype(np.float32)

    window = np.hanning(n_fft).astype(np.float64)
    y = np.asarray(y, dtype=np.float64)

    # STFT
    n_frames = 1 + (len(y) - n_fft) // hop
    if n_frames < 2:
        return y.astype(np.float32)
    idx = np.arange(n_fft)[None, :] + hop * np.arange(n_frames)[:, None]
    frames = y[idx] * window
    stft = np.fft.rfft(frames, axis=1)  # (n_frames, n_bins)

    mag = np.abs(stft)
    phase = np.angle(stft)
    bins = stft.shape[1]

    # phases attendues entre trames (avance nominale)
    omega = 2.0 * np.pi * hop * np.arange(bins) / n_fft

    # ré-échantillonne les trames aux positions temporelles étirées
    time_steps = np.arange(0, n_frames - 1, rate)
    out_frames = len(time_steps)

    acc_phase = phase[0].copy()
    out_stft = np.zeros((out_frames, bins), dtype=np.complex128)
    for i, t in enumerate(time_steps):
        k = int(np.floor(t))
        frac = t - k
        # interpolation linéaire de la magnitude
        m = (1.0 - frac) * mag[k] + frac * mag[min(k + 1, n_frames - 1)]
        out_stft[i] = m * np.exp(1j * acc_phase)
        # avance de phase = delta mesuré ramené dans (-pi, pi] + avance nominale
        dphi = phase[min(k + 1, n_frames - 1)] - phase[k] - omega
        dphi = dphi - 2.0 * np.pi * np.round(dphi / (2.0 * np.pi))
        acc_phase = acc_phase + omega + dphi

    # iSTFT overlap-add
    out_len = n_fft + hop * (out_frames - 1)
    out = np.zeros(out_len, dtype=np.float64)
    wsum = np.zeros(out_len, dtype=np.float64)
    inv = np.fft.irfft(out_stft, n=n_fft, axis=1) * window
    for i in range(out_frames):
        s = i * hop
        out[s:s + n_fft] += inv[i]
        wsum[s:s + n_fft] += window ** 2
    out = out / np.maximum(wsum, 1e-8)
    m = np.max(np.abs(out))
    if m > 1e-8:
        out = out / m * min(1.0, np.max(np.abs(y)) if np.max(np.abs(y)) > 0 else 1.0)
    return out.astype(np.float32)


def pink_noise(n: int, rng: np.random.Generator) -> np.ndarray:
    """Bruit rose (spectre en 1/f), plus réaliste qu'un bruit blanc."""
    white = rng.standard_normal(n)
    spec = np.fft.rfft(white)
    freqs = np.arange(len(spec))
    freqs[0] = 1
    spec = spec / np.sqrt(freqs)
    out = np.fft.irfft(spec, n=n)
    out = out / (np.max(np.abs(out)) + 1e-9)
    return out.astype(np.float32)


def add_noise(y: np.ndarray, snr_db: float, rng: np.random.Generator, kind: str = "pink") -> np.ndarray:
    noise = pink_noise(len(y), rng) if kind == "pink" else rng.standard_normal(len(y)).astype(np.float32)
    sig_p = np.mean(y ** 2) + 1e-12
    noise_p = np.mean(noise ** 2) + 1e-12
    target_noise_p = sig_p / (10.0 ** (snr_db / 10.0))
    noise = noise * np.sqrt(target_noise_p / noise_p)
    return (y + noise).astype(np.float32)


def random_gain(y: np.ndarray, rng: np.random.Generator, low_db: float = -12.0, high_db: float = 3.0) -> np.ndarray:
    g = 10.0 ** (rng.uniform(low_db, high_db) / 20.0)
    return (y * g).astype(np.float32)


def place_in_window(y: np.ndarray, total: int, rng: np.random.Generator,
                    keep_onset: bool = False) -> np.ndarray:
    """Place ``y`` (le token) dans une fenêtre de ``total`` échantillons.

    ``keep_onset=True`` garantit que le DEBUT du token (l'attaque consonantique) est
    conservé : indispensable pour les notes, dont l'onset (d/r/m/f/s/l) distingue
    ``do`` de ``sol`` (même voyelle tenue « ooo »). ``keep_onset=False`` autorise un
    recadrage quelconque (utile pour la classe _noise_).
    """
    if len(y) >= total:
        if keep_onset:
            jitter = int(0.02 * total)  # petit décalage <= 20 ms
            start = int(rng.integers(0, jitter + 1))
        else:
            start = int(rng.integers(0, len(y) - total + 1))
        return y[start:start + total].astype(np.float32)
    out = np.zeros(total, dtype=np.float32)
    max_off = total - len(y)
    offset = int(rng.integers(0, max_off + 1))
    out[offset:offset + len(y)] = y
    return out


def clip(y: np.ndarray) -> np.ndarray:
    return np.clip(y, -1.0, 1.0).astype(np.float32)
