# Pipeline & exploitation

## Arborescence
```
poc--finetune--tts-solfege/
├── CLAUDE.md               # contexte agent (référence agent/* via @)
├── README.md               # présentation lisible sur GitHub
├── config.json             # SOURCE UNIQUE : SR, STFT, mel, labels (Python + JS)
├── requirements.txt        # numpy scipy torch(CPU) onnx onnxruntime pytest
├── pytest.ini
├── solfege/                # bibliothèque Python
│   ├── features.py         # log-mel (numpy pur) — miroir de web/features.js
│   ├── tts.py              # génération espeak-ng (notes + parasites)
│   ├── augment.py          # vocodeur de phase (tenues), bruit, gain, placement
│   ├── dataset.py          # pool TTS, split par rendu, build_xy (clean/hard)
│   ├── model.py            # CNN (pooling avg⊕max)
│   ├── train.py            # entraînement + export ONNX + metrics.json
│   └── infer.py            # classification + décodage en flux
├── scripts/
│   ├── make_fixtures.py    # vecteurs de référence parité JS
│   ├── gen_samples.py      # extraits .wav de démo
│   ├── export_standalone.py + standalone_template.html  # page HTML auto-contenue
│   ├── dump_verify_cases.py + verify_standalone.mjs      # vérif JS pur == PyTorch
│   └── (web/nn.js = forward CNN en JS pur)
├── tests/
│   ├── test_features.py    # features + parité Python↔JS (node)
│   ├── test_tts.py         # espeak dispo, synthèse, vitesse
│   ├── test_accuracy.py    # >= 99% (TTS frais), notes, parasites, tenues, flux
│   ├── js_features_check.mjs
│   └── fixtures/feature_parity.json
├── web/                    # démo statique (publiée sur .pages.dev)
│   ├── index.html · style.css · app.js · features.js
│   ├── model.onnx · labels.json          # modèle livré (committé)
│   └── samples/*.wav + index.json        # extraits de test
├── artifacts/metrics.json  # accuracy clean/hard/voix inconnues + confusions
├── wrangler.toml
└── .github/workflows/      # ci.yml (tests), deploy.yml (Cloudflare), pages.yml (fallback)
```

## Entraîner / régénérer
```bash
sudo apt-get install -y espeak-ng
pip install -r requirements.txt
python -m solfege.train --n-per-class 1800 --epochs 30    # -> web/model.onnx + artifacts/metrics.json
python scripts/make_fixtures.py                            # si features.py change
python scripts/gen_samples.py                              # si on veut d'autres extraits
```
Le pool TTS est re-synthétisé à chaque entraînement (déterministe par graine).
`web/model.onnx`, `web/labels.json` et `web/samples/` sont **committés**.

## Tester
```bash
pytest                       # features + parité JS + TTS + accuracy >= 99%
pytest --runslow             # ajoute un ré-entraînement court qui doit atteindre la cible
```
La CI (`ci.yml`) installe espeak-ng, PyTorch CPU, puis lance ces tests à chaque push.

## Prévisualiser la démo
```bash
python -m http.server -d web 8000    # http://localhost:8000 (micro nécessite localhost/https)
```

## Déployer sur `.pages.dev`
```bash
npx wrangler pages deploy web --project-name solfege-stt
```
Ou automatique via `deploy.yml` (secrets `CLOUDFLARE_API_TOKEN` + `CLOUDFLARE_ACCOUNT_ID`
dans *Settings → Secrets*). Fallback GitHub Pages : `pages.yml` (déclenchement manuel).

## Invariants à préserver
1. `config.json` = seule source des paramètres audio ; features.py et features.js d'accord (test).
2. Le modèle livré (`web/model.onnx`) correspond à `config.json`/`labels.json`.
3. `pytest` reste vert, notamment le seuil de 99 % sur TTS frais.
4. La démo tourne **entièrement dans le navigateur** (pas de serveur d'inférence).
