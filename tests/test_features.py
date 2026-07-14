"""Tests de l'extraction de features + parité Python<->JS."""
import json
import os
import shutil
import subprocess

import numpy as np
import pytest

from solfege.features import (feature_stack, load_config, logmel, n_channels,
                              n_frames)

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
FIXTURE = os.path.join(ROOT, "tests", "fixtures", "feature_parity.json")


def test_shape():
    cfg = load_config()
    y = np.zeros(cfg["window_samples"], dtype=np.float32)
    assert logmel(y, cfg).shape == (cfg["mel"]["n_mels"], n_frames(cfg))
    assert feature_stack(y, cfg).shape == (n_channels(), cfg["mel"]["n_mels"], n_frames(cfg))


def test_gain_invariance():
    """La normalisation par moyenne rend les features quasi invariantes au gain.

    Vraie pour un signal plein (chaque bande mel se décale de 2·log(g), constante
    retirée par la soustraction de moyenne). Le résiduel vient du plancher 1e-6 sur
    les bandes de faible énergie — négligeable.
    """
    cfg = load_config()
    rng = np.random.default_rng(0)
    y = rng.standard_normal(cfg["window_samples"]).astype(np.float32) * 0.3
    a = logmel(y)
    b = logmel(y * 0.25)  # -12 dB
    assert np.max(np.abs(a - b)) < 0.05
    assert np.median(np.abs(a - b)) < 1e-2


def test_python_reproduces_fixtures():
    assert os.path.exists(FIXTURE), "lancez scripts/make_fixtures.py"
    fx = json.load(open(FIXTURE, encoding="utf-8"))
    for c in fx["cases"]:
        feat = feature_stack(np.array(c["waveform"], dtype=np.float32))
        assert list(feat.shape) == c["shape"]
        ref = np.array(c["features"], dtype=np.float32).reshape(c["shape"])
        assert np.max(np.abs(feat - ref)) < 1e-4, f"cas {c['name']}"


@pytest.mark.skipif(shutil.which("node") is None, reason="node absent")
def test_js_python_parity():
    """web/features.js doit produire les mêmes features que Python (parité navigateur)."""
    assert os.path.exists(FIXTURE), "lancez scripts/make_fixtures.py"
    script = os.path.join(ROOT, "tests", "js_features_check.mjs")
    res = subprocess.run(["node", script], capture_output=True, text=True)
    assert res.returncode == 0, f"parité JS échouée:\n{res.stdout}\n{res.stderr}"
