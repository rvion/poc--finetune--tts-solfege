"""Tests du générateur TTS (espeak-ng)."""
import numpy as np
import pytest

from solfege.tts import (NOTE_TEXT, TTSParams, espeak_available, synth,
                         trim_silence)

pytestmark = pytest.mark.skipif(not espeak_available(),
                                reason="espeak-ng absent (apt-get install espeak-ng)")


def test_synth_returns_audio():
    y = synth("do", TTSParams("fr-fr", 160, 50))
    assert y.dtype == np.float32
    assert len(y) > 1000
    assert np.max(np.abs(y)) > 0.01  # signal non silencieux


def test_all_notes_synthesizable():
    for label, text in NOTE_TEXT.items():
        y = trim_silence(synth(text, TTSParams("fr-fr", 150, 50)))
        assert len(y) > 200, f"note {label} vide"


def test_speed_affects_duration():
    """Une vitesse plus élevée doit raccourcir la même syllabe."""
    slow = trim_silence(synth("sol", TTSParams("fr-fr", 90, 50)))
    fast = trim_silence(synth("sol", TTSParams("fr-fr", 320, 50)))
    assert len(fast) < len(slow)


def test_resampled_to_target_sr():
    from solfege.features import load_config
    sr = load_config()["sample_rate"]
    y = synth("mi", TTSParams("fr-fr", 160, 50), target_sr=sr)
    # ~ moins de 2 s de signal
    assert len(y) < 2 * sr
