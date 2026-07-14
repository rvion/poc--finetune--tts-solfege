# CLAUDE.md — contexte de l'agent

POC : un **classifieur audio compact** qui reconnaît les notes de solfège
**do ré mi fa sol la si** — parlées, tenues (« dooooo ») ou rapides — et rejette
tout le reste (« euh… », « ah non… ») comme du **bruit**. Données 100 % synthétiques
(TTS espeak-ng), modèle exporté en ONNX, **démo en ligne** dans le navigateur.

## Spécification (spec-first, chargée automatiquement)
- @agent/00-vision.md — vision, principes, définition de « terminé »
- @agent/01-besoins.md — besoins, hypothèses, non-objectifs
- @agent/02-specification.md — spécification technique et livrables
- @agent/03-decisions.md — journal des décisions (ADR léger)
- @agent/04-pipeline.md — arborescence, entraînement, tests, déploiement

## Règles d'or
1. **`config.json` est la source unique de vérité** des paramètres audio (SR, STFT,
   mel, labels). Python (`solfege/features.py`) et JS (`web/features.js`) le lisent.
2. **Parité features Python ↔ JS obligatoire.** Toute modification de l'extraction
   doit garder `pytest tests/test_features.py` vert (tolérance 1e-4). Régénérer les
   vecteurs de référence : `python scripts/make_fixtures.py`.
3. **Le modèle livré vit dans `web/`** (`model.onnx` + `labels.json`), committé pour
   que la démo marche sans ré-entraîner. Le régénérer : `python -m solfege.train`.
4. **Cible : ≥ 99 % d'accuracy** sur des notes articulées TTS (rendus inédits), vérifié
   par `pytest tests/test_accuracy.py`. Deux métriques secondaires reportées : forte
   augmentation (bruit) et voix jamais vues.
5. **Le TTS teste le STT.** Les tests re-synthétisent un jeu d'évaluation frais avec
   espeak-ng — jamais de dépendance à un modèle figé non vérifié.

## Commandes utiles
```bash
sudo apt-get install -y espeak-ng                 # moteur TTS (obligatoire)
pip install -r requirements.txt                   # numpy scipy torch(CPU) onnx onnxruntime pytest
python -m solfege.train --n-per-class 1800 --epochs 30   # entraîne + exporte web/model.onnx
python scripts/make_fixtures.py                   # régénère les vecteurs de parité JS
python scripts/gen_samples.py                     # régénère les clips de démo web/samples/
pytest                                            # features + TTS + accuracy >= 99%
python -m http.server -d web 8000                 # prévisualise la démo (http://localhost:8000)
npx wrangler pages deploy web --project-name solfege-stt   # déploie sur .pages.dev
```
