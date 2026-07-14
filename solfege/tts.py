"""Génération de données audio par TTS (espeak-ng).

espeak-ng est un moteur *text-to-speech* léger et déterministe. On l'utilise pour
fabriquer un corpus varié de notes de solfège chantées/parlées ainsi que de mots
« parasites » (euh, ah non, ...) — sans jamais avoir à enregistrer de vraie voix.

Le même moteur sert à générer le jeu d'ENTRAINEMENT et, dans les tests, un jeu
d'EVALUATION frais et disjoint : « utiliser du TTS pour tester le STT ».
"""
from __future__ import annotations

import shutil
import subprocess
import tempfile
import os
from dataclasses import dataclass

import numpy as np
from scipy.io import wavfile
from scipy.signal import resample_poly

from .features import load_config

# Voix espeak-ng qui prononcent correctement les syllabes de solfège (langues latines
# + anglais pour la diversité de timbre). Toutes présentes dans le paquet espeak-ng.
VOICES = ["fr-fr", "fr-be", "fr-ch", "it", "es", "ca", "pt", "ro", "en-us"]

# Écriture par voix pour obtenir la bonne prononciation de chaque note.
# Clé = label ascii ; valeur = texte passé à espeak (accent pour "ré").
NOTE_TEXT = {
    "do": "do",
    "re": "ré",
    "mi": "mi",
    "fa": "fa",
    "sol": "sol",
    "la": "la",
    "si": "si",
}

# Mots/onomatopées parasites (classe _noise_). Ce sont des sons CLAIREMENT distincts
# des notes — hésitations, mots courants, onomatopées. On évite volontairement les
# *homophones* de notes (« mie »≈mi, « scie »≈si, « la »/« lala », « dodo », « fa la »…) :
# demander de les classer en « bruit » serait contradictoire (ils sonnent comme des
# notes) et polluerait l'apprentissage des deux côtés.
GARBAGE_WORDS = [
    "euh", "euuuh", "ah non", "ah", "hmm", "hein", "bah", "ben", "voilà", "attends",
    "pardon", "bonjour", "merci", "oui", "non", "peut-être", "je sais pas", "alors",
    "donc", "quoi", "ok", "d'accord", "stop", "encore", "et", "un", "deux", "trois",
    "note", "chante", "gamme", "musique", "faux", "juste", "tiens", "oups", "aïe",
    "salut", "coucou", "ouais", "bof", "hop", "allez", "super", "génial", "zut",
    "comment", "pourquoi", "attention", "écoute", "regarde", "papa", "maman",
    "table", "chaise", "chien", "chat", "rouge", "bleu", "vite", "lent",
]


@dataclass
class TTSParams:
    voice: str = "fr-fr"
    speed: int = 160       # mots/min (80 = très lent/tenu, 350 = très rapide)
    pitch: int = 50        # 0-99
    amplitude: int = 100   # 0-200
    gap: int = 0           # pause entre mots (unités de 10 ms)


def espeak_available() -> bool:
    return shutil.which("espeak-ng") is not None or shutil.which("espeak") is not None


def _binary() -> str:
    return shutil.which("espeak-ng") or shutil.which("espeak")


def synth(text: str, params: TTSParams, target_sr: int | None = None) -> np.ndarray:
    """Synthétise ``text`` et retourne un waveform float32 mono [-1, 1] à ``target_sr``."""
    cfg = load_config()
    target_sr = target_sr or cfg["sample_rate"]
    binary = _binary()
    if binary is None:
        raise RuntimeError("espeak-ng introuvable : installez-le (apt-get install espeak-ng).")

    with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as tmp:
        path = tmp.name
    try:
        cmd = [
            binary, "-v", params.voice,
            "-s", str(params.speed),
            "-p", str(params.pitch),
            "-a", str(params.amplitude),
            "-g", str(params.gap),
            "-w", path, text,
        ]
        subprocess.run(cmd, check=True, capture_output=True)
        sr, data = wavfile.read(path)
    finally:
        if os.path.exists(path):
            os.remove(path)

    if data.dtype == np.int16:
        y = data.astype(np.float32) / 32768.0
    elif data.dtype == np.int32:
        y = data.astype(np.float32) / 2147483648.0
    else:
        y = data.astype(np.float32)
    if y.ndim > 1:
        y = y.mean(axis=1)

    if sr != target_sr:
        from math import gcd
        g = gcd(sr, target_sr)
        y = resample_poly(y, target_sr // g, sr // g).astype(np.float32)
    return y


def trim_silence(y: np.ndarray, thresh: float = 0.01) -> np.ndarray:
    """Retire le silence de tête/queue (seuil sur l'amplitude absolue)."""
    mask = np.abs(y) > thresh
    if not mask.any():
        return y
    i0, i1 = np.argmax(mask), len(mask) - np.argmax(mask[::-1])
    return y[i0:i1]
