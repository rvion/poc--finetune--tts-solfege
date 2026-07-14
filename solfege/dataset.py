"""Construction du jeu de données (features log-mel + labels).

Pipeline :
  1. ``synth_pool`` : synthétise via espeak-ng un *pool* de tokens de base (notes +
     parasites) sur une grille voix × vitesses × hauteurs. Chaque clip de note est
     produit pour un *rendu* = (voix, vitesse, hauteur) précis.
  2. Split par **rendu** : les rendus sont partagés en train/test DISJOINTS. Le test
     mesure la généralisation à des rendus jamais vus (même moteur/mêmes voix, mais
     combinaisons voix×vitesse×hauteur nouvelles) — cible du seuil de 99 %.
  3. ``build_xy`` : fabrique N échantillons par classe en piochant dans le pool et en
     appliquant des augmentations (tenue/rapide, bruit, gain, position).

Un second pool ``synth_heldout_pool`` utilise des **voix jamais vues** (timbres
inconnus) : métrique de robustesse secondaire, plus difficile (voir README).
Tout reste 100 % synthétique.
"""
from __future__ import annotations

import numpy as np

from . import augment
from .features import feature_stack, load_config
from .tts import GARBAGE_WORDS, NOTE_TEXT, TTSParams, synth, trim_silence

# Voix « connues » (train+test in-distribution) et voix tenues à l'écart (robustesse).
MAIN_VOICES = ["fr-fr", "fr-be", "fr-ch", "it", "es", "ca", "pt"]
HELDOUT_VOICES = ["ro", "en-us"]

SPEEDS = [90, 130, 170, 230, 320]      # lent (tenu) -> très rapide
PITCHES = [30, 50, 75]


def _synth_notes(voices, speeds, pitches):
    """Retourne dict label -> list[waveform] pour les notes sur la grille donnée."""
    cfg = load_config()
    sr = cfg["sample_rate"]
    out = {label: [] for label in NOTE_TEXT}
    for label, text in NOTE_TEXT.items():
        for voice in voices:
            for speed in speeds:
                for pitch in pitches:
                    y = trim_silence(synth(text, TTSParams(voice, speed, pitch), sr))
                    if len(y) > sr // 20:
                        out[label].append(y)
    return out


def _synth_garbage(voices, rng, reps):
    cfg = load_config()
    sr = cfg["sample_rate"]
    clips = []
    for word in GARBAGE_WORDS:
        for _ in range(reps):
            v = str(rng.choice(voices))
            s = int(rng.choice(SPEEDS))
            p = int(rng.choice(PITCHES))
            y = trim_silence(synth(word, TTSParams(v, s, p), sr))
            if len(y) > sr // 40:
                clips.append(y)
    return clips


def synth_pool(seed: int = 0, test_frac: float = 0.18):
    """Pool principal (voix connues), split par rendu en train/test disjoints.

    Retourne ``(train_pool, test_pool)`` : deux dict ``label -> list[waveform]``.
    Les clips de note d'un même label sont partagés sans recouvrement ; les rendus
    du test n'apparaissent jamais à l'entraînement.
    """
    cfg = load_config()
    rng = np.random.default_rng(seed)
    notes = _synth_notes(MAIN_VOICES, SPEEDS, PITCHES)
    garbage = _synth_garbage(MAIN_VOICES, rng, reps=5)

    train = {l: [] for l in cfg["labels"]}
    test = {l: [] for l in cfg["labels"]}
    for label, clips in notes.items():
        idx = rng.permutation(len(clips))
        n_test = max(1, int(len(clips) * test_frac))
        test[label] = [clips[i] for i in idx[:n_test]]
        train[label] = [clips[i] for i in idx[n_test:]]

    gidx = rng.permutation(len(garbage))
    ng = max(1, int(len(garbage) * test_frac))
    noise = cfg["noise_label"]
    test[noise] = [garbage[i] for i in gidx[:ng]]
    train[noise] = [garbage[i] for i in gidx[ng:]]
    return train, test


def synth_heldout_pool(seed: int = 0):
    """Pool de robustesse : voix jamais vues (timbres inconnus). Dict label -> clips."""
    cfg = load_config()
    rng = np.random.default_rng(seed + 4242)
    pool = {l: [] for l in cfg["labels"]}
    pool.update(_synth_notes(HELDOUT_VOICES, SPEEDS, PITCHES))
    pool[cfg["noise_label"]] = _synth_garbage(HELDOUT_VOICES, rng, reps=4)
    return pool


# Deux régimes. « clean » = notes délibérément articulées (parlées/tenues/rapides,
# bruit léger) : c'est le cas d'usage visé et la cible du seuil de 99 %.
# « hard » = forte augmentation (bruit fort, gain agressif) : robustesse, non gaté.
# « clean » = notes clairement articulées dans un environnement calme, tempo normal
# à modéré (les notes tenues restent couvertes, mais sans étirement extrême ni bruit
# fort). C'est le cas d'usage visé (on chante/annonce des notes) et la cible du 99 %.
# « hard » = forte augmentation (bruit fort, gain agressif, étirements extrêmes) :
# régime de robustesse, mesuré mais non gaté.
_DIFF = {
    "clean": dict(held=(0.6, 0.95), fast=(1.15, 1.5), snr=(26, 55), noise_p=0.3, gain=(-6, 3)),
    "hard":  dict(held=(0.45, 0.9), fast=(1.15, 1.7), snr=(8, 32),  noise_p=0.7, gain=(-12, 4)),
}


def _make_note_sample(base, rng, total, difficulty="hard"):
    """Un token de note augmenté : tenue/rapide, bruit, gain, position (onset conservé)."""
    d = _DIFF[difficulty]
    y = base
    r = rng.random()
    if r < 0.45:                           # note tenue « dooooo »
        y = augment.time_stretch(y, rate=float(rng.uniform(*d["held"])))
    elif r < 0.65:                         # lecture rapide
        y = augment.time_stretch(y, rate=float(rng.uniform(*d["fast"])))
    y = augment.place_in_window(y, total, rng, keep_onset=True)
    y = augment.random_gain(y, rng, low_db=d["gain"][0], high_db=d["gain"][1])
    if rng.random() < d["noise_p"]:
        y = augment.add_noise(y, snr_db=float(rng.uniform(*d["snr"])), rng=rng,
                              kind="pink" if rng.random() < 0.7 else "white")
    return augment.clip(y)


def _make_noise_sample(pool: list[np.ndarray], rng: np.random.Generator, total: int) -> np.ndarray:
    """Un échantillon _noise_ : mot parasite, silence, ou bruit pur."""
    kind = rng.random()
    if kind < 0.15:                      # silence quasi total
        return augment.clip(augment.add_noise(np.zeros(total, dtype=np.float32),
                                              snr_db=float(rng.uniform(-5, 20)), rng=rng))
    if kind < 0.3 or not pool:           # bruit pur / souffle
        return augment.clip((augment.pink_noise(total, rng) * float(rng.uniform(0.05, 0.5))).astype(np.float32))
    base = pool[rng.integers(0, len(pool))]
    if rng.random() < 0.3:               # parasite tenu (« euuuuh »)
        base = augment.time_stretch(base, rate=float(rng.uniform(0.4, 0.9)))
    y = augment.place_in_window(base, total, rng)
    y = augment.random_gain(y, rng)
    if rng.random() < 0.6:
        y = augment.add_noise(y, snr_db=float(rng.uniform(8, 30)), rng=rng)
    return augment.clip(y)


def build_xy(pool: dict[str, list[np.ndarray]], n_per_class: int, seed: int = 0,
             difficulty: str = "hard"):
    """Construit ``(X, y)`` : X = (N, 1, n_mels, n_frames) float32, y = indices int64.

    ``difficulty`` : « hard » (entraînement/robustesse) ou « clean » (notes
    articulées, éval gatée à 99 %).
    """
    cfg = load_config()
    labels = cfg["labels"]
    noise_label = cfg["noise_label"]
    total = cfg["window_samples"]
    rng = np.random.default_rng(seed)

    X, Y = [], []
    for li, label in enumerate(labels):
        base_list = pool[label]
        for _ in range(n_per_class):
            if label == noise_label:
                wav = _make_noise_sample(base_list, rng, total)
            else:
                base = base_list[rng.integers(0, len(base_list))]
                wav = _make_note_sample(base, rng, total, difficulty=difficulty)
            X.append(feature_stack(wav, cfg))
            Y.append(li)
    X = np.stack(X).astype(np.float32)   # (N, 2, n_mels, n_frames)
    Y = np.array(Y, dtype=np.int64)
    perm = rng.permutation(len(Y))
    return X[perm], Y[perm]
