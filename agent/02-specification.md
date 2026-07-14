# Spécification

## Cadrage du problème
Reconnaissance de **8 classes** sur une fenêtre audio de 1 s :
`do, ré, mi, fa, sol, la, si` + `_noise_`. C'est du **keyword spotting** (petit
vocabulaire), pas de l'ASR généraliste. Le `do` aigu de fin de gamme = même classe `do`.

## Chaîne de traitement
```
audio 16 kHz ──► log-mel (40 × 98) ──► CNN (~150k params) ──► softmax 8 classes
   fenêtre 1 s      features.py / .js        model.onnx           argmax + seuil
```

### Audio & features (`config.json`, source unique)
- Échantillonnage **16 kHz**, fenêtre **1,0 s** (16000 échantillons).
- Pré-accentuation 0,97 ; STFT `n_fft=512`, fenêtre 400 (25 ms), pas 160 (10 ms), Hann.
- Banc de filtres **mel HTK** 40 bandes, 40–8000 Hz ; `log(mel + 1e-6)`.
- Normalisation par énoncé = **soustraction de la moyenne** (invariance au gain ; pas de
  division par l'écart-type, instable sur le silence).
- Sortie **(40, 98)**. Implémentée en `solfege/features.py` **et** `web/features.js`
  (parité 1e-4, testée).

### Génération de données (TTS, `solfege/tts.py` + `dataset.py`)
- espeak-ng, voix latines + anglais, vitesses 90→320 (lent/tenu → rapide), hauteurs variées.
- **Notes tenues** : vocodeur de phase (`augment.time_stretch`, rate 0,45–0,9) qui allonge
  la voyelle sans changer la hauteur ; l'**attaque consonantique reste dans la fenêtre**
  (sinon « ooo » ambigu entre do et sol).
- **Classe `_noise_`** : mots parasites (« euh », « ah non », …), quasi-homophones
  distracteurs, silence, bruit rose/blanc.
- **Augmentation** : bruit additif (SNR variable), gain, position aléatoire.
- **Split par rendu** : les combinaisons (voix × vitesse × hauteur) du test sont
  disjointes de l'entraînement → test de généralisation, pas de mémorisation.

### Modèle (`solfege/model.py`)
- CNN 4 blocs conv (32→64→96→96) + **pooling global moyenne ⊕ maximum**. Le max
  préserve la trace de l'onset bref (d/s/m/f/l) qui sépare des voyelles tenues identiques.
- Tête MLP → 8 logits. Export **ONNX opset 13**, batch dynamique.

### Décodage en flux (`solfege/infer.py` + `web/app.js`)
- Fenêtre glissante (pas 250 ms) ; une prédiction par fenêtre.
- Fusion des prédictions consécutives identiques + **suppression de `_noise_`** →
  suite de notes. Seuil de confiance réglable.

## Livrables
| Sortie | Rôle |
|---|---|
| `config.json` | paramètres audio + labels (source unique Python/JS) |
| `solfege/` | features, tts, augment, dataset, model, train, infer |
| `web/model.onnx`, `web/labels.json` | modèle livré, chargé par la démo |
| `web/index.html` + `app.js` + `features.js` + `style.css` | démo navigateur |
| `web/samples/*.wav` | extraits de test (générés) |
| `tests/` | features+parité JS, TTS, **accuracy ≥ 99 %**, décodage, notes tenues |
| `artifacts/metrics.json` | accuracy clean / hard / voix inconnues + matrices de confusion |
| `wrangler.toml`, `.github/workflows/` | déploiement `.pages.dev` + CI |

## Critères de validation
- `pytest tests/test_accuracy.py` : ≥ 99 % (notes articulées, rendus TTS inédits),
  rappel notes ≥ 99 %, parasites rejetés, notes tenues reconnues, gamme décodée dans l'ordre.
- `pytest tests/test_features.py` : parité Python ↔ JS (1e-4).
- Métriques secondaires (reportées, non gatées) : robustesse forte augmentation, voix inconnues.
