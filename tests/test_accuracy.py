"""Test central : le modèle livré atteint >= 99 % sur un jeu TTS frais.

« Utiliser du text-to-speech pour tester le speech-to-text » : on RE-synthétise
un jeu d'évaluation avec espeak-ng (rendus jamais vus à l'entraînement) et on
vérifie que le modèle committé (``web/model.onnx``) dépasse le seuil de config.
"""
import os

import numpy as np
import pytest

from solfege.dataset import build_xy, synth_pool
from solfege.features import load_config
from solfege.tts import TTSParams, espeak_available, synth, trim_silence

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
ONNX = os.path.join(ROOT, "web", "model.onnx")

pytestmark = pytest.mark.skipif(not espeak_available(), reason="espeak-ng absent")


def _classifier():
    pytest.importorskip("onnxruntime")
    if not os.path.exists(ONNX):
        pytest.skip("web/model.onnx manquant : lancez `python -m solfege.train`")
    from solfege.infer import OnnxClassifier
    return OnnxClassifier(ONNX)


@pytest.fixture(scope="module")
def clf():
    return _classifier()


def test_accuracy_above_target(clf):
    """Précision globale (7 notes + bruit) >= seuil de config sur des rendus TTS inédits."""
    cfg = load_config()
    target = cfg["accuracy_target"]
    _, test_pool = synth_pool(seed=123)          # rendus disjoints de l'entraînement
    X, Y = build_xy(test_pool, n_per_class=180, seed=777, difficulty="clean")
    logits = clf.logits(X)                        # X: (N,1,n_mels,n_frames) -> (N, n_classes)
    pred = logits.argmax(1)
    acc = float((pred == Y).mean())
    print(f"\naccuracy globale (notes articulées) = {acc*100:.2f}%  (cible {target*100:.0f}%)")
    assert acc >= target, f"{acc*100:.2f}% < cible {target*100:.0f}%"


def test_notes_recall(clf):
    """Chaque note isolée, clairement articulée, est correctement reconnue (rappel >= 99 %)."""
    cfg = load_config()
    labels = cfg["labels"]
    notes = cfg["note_labels"]
    voices = ["fr-fr", "fr-be", "it", "es"]
    correct = total = 0
    per = {n: [0, 0] for n in notes}
    from solfege.tts import NOTE_TEXT
    for n in notes:
        for v in voices:
            for speed in (120, 200):
                y = trim_silence(synth(NOTE_TEXT[n], TTSParams(v, speed, 50)))
                label, _, _ = clf.classify(y)
                ok = (label == n)
                correct += ok; total += 1
                per[n][0] += ok; per[n][1] += 1
    print("\nrappel par note :", {k: f"{v[0]}/{v[1]}" for k, v in per.items()})
    assert correct / total >= 0.99, f"rappel notes {correct}/{total}"


def test_garbage_rejected(clf):
    """Les mots parasites et le silence sont classés comme _noise_ (non-notes)."""
    cfg = load_config()
    noise = cfg["noise_label"]
    words = ["euh", "ah non", "bonjour", "attends", "voilà", "pardon", "hmm"]
    rejected = 0
    for w in words:
        y = trim_silence(synth(w, TTSParams("fr-fr", 160, 50)))
        label, conf, _ = clf.classify(y)
        rejected += (label == noise)
    # silence pur
    sil, _, _ = clf.classify(np.zeros(cfg["window_samples"], dtype=np.float32))
    assert sil == noise
    assert rejected >= len(words) - 1, f"{rejected}/{len(words)} parasites rejetés"


def test_streaming_decode_scale(clf):
    """Décodage en flux d'une gamme claire -> récupère les notes dans l'ordre."""
    from solfege.infer import stream_decode
    cfg = load_config()
    sr = cfg["sample_rate"]
    from solfege.tts import NOTE_TEXT
    notes = cfg["note_labels"]
    clips = [trim_silence(synth(NOTE_TEXT[n], TTSParams("fr-fr", 150, 50))) for n in notes]
    gap = np.zeros(int(0.4 * sr), dtype=np.float32)
    wav = np.concatenate([c for pair in zip(clips, [gap] * len(clips)) for c in pair])
    seq = stream_decode(clf, wav, hop_sec=0.2, threshold=0.55)
    # sous-séquence : les notes correctes apparaissent dans le bon ordre
    matched, j = 0, 0
    for note in notes:
        while j < len(seq) and seq[j] != note:
            j += 1
        if j < len(seq):
            matched += 1; j += 1
    print(f"\ngamme décodée = {seq}")
    assert matched >= 6, f"seulement {matched}/7 notes retrouvées dans l'ordre : {seq}"


def test_held_note_recognized(clf):
    """Une note TENUE (« dooooo ») reste correctement classée."""
    from solfege.augment import time_stretch
    from solfege.tts import NOTE_TEXT
    ok = 0
    targets = ["do", "re", "mi", "sol", "la"]
    for n in targets:
        base = trim_silence(synth(NOTE_TEXT[n], TTSParams("fr-fr", 140, 50)))
        held = time_stretch(base, 0.5)  # ~2x plus long
        label, _, _ = clf.classify(held)
        ok += (label == n)
    assert ok >= len(targets) - 1, f"notes tenues reconnues : {ok}/{len(targets)}"


@pytest.mark.slow
def test_training_reaches_target():
    """Reproductibilité : un entraînement court atteint le seuil (lancer --runslow)."""
    from solfege.train import train
    _, acc, _ = train(n_per_class=500, epochs=18, seed=0, out="artifacts_test", quiet=True)
    assert acc >= load_config()["accuracy_target"] - 0.01
