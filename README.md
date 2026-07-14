# 🎵 Solfège STT — reconnaître *do ré mi fa sol la si* (TTS → STT)

POC : à partir d'un **modèle de synthèse vocale** (TTS, *espeak-ng*), on fabrique un corpus
de notes de solfège, puis on entraîne un **classifieur / speech-to-text compact** qui
reconnaît **do · ré · mi · fa · sol · la · si** — qu'elles soient **parlées**, **tenues**
(« dooooo ») ou **rapides** — et qui **ignore tout le reste** (« euh… », « ah non… »,
silence, bruit) comme du *bruit inintéressant*, pour ne garder que la suite de notes.

Le modèle tourne **dans le navigateur** (ONNX, temps réel au micro) et se déploie sur
`*.pages.dev` (Cloudflare Pages / Wrangler).

> **Rien n'est enregistré à la voix humaine** : entraînement **100 % synthétique**.
> Et — c'est l'idée forte — **le TTS sert aussi à tester le STT** : les tests
> re-synthétisent un jeu d'évaluation frais et vérifient automatiquement le seuil de 99 %.

---

## En bref

| | |
|---|---|
| **Entrée** | audio micro / `.wav`, fenêtre 1 s, 16 kHz |
| **Sortie** | une des 8 classes : `do ré mi fa sol la si` + `bruit`, puis suite de notes |
| **Features** | log-mel 40×98, identiques en Python et en JS (parité testée à 1e-4) |
| **Modèle** | CNN ~150k paramètres, export ONNX (~0,6 Mo) |
| **Données** | espeak-ng, 7 voix, vitesses 90→320, hauteurs variées, notes tenues + parasites |
| **Cible** | **≥ 99 %** sur notes articulées (rendus TTS inédits), vérifié par `pytest` |
| **Démo** | 100 % navigateur (onnxruntime-web), déployée sur `.pages.dev` |

---

## Démarrage rapide

```bash
# 1. dépendances
sudo apt-get install -y espeak-ng
pip install -r requirements.txt

# 2. entraîner (re-synthétise le corpus, exporte web/model.onnx)
python -m solfege.train --n-per-class 1800 --epochs 30

# 3. tester (features + parité JS + TTS + accuracy >= 99%)
pytest

# 4. démo locale
python -m http.server -d web 8000     # http://localhost:8000
```

Un modèle est **déjà committé** dans `web/` : la démo et les tests fonctionnent sans
ré-entraîner.

---

## Comment ça marche

```
        espeak-ng (TTS)                        entraînement                inférence (navigateur)
  ┌──────────────────────┐   log-mel   ┌───────────────────┐   ONNX   ┌────────────────────────┐
  │ "do" "ré" … "si"      │──────────►  │  CNN 8 classes    │ ───────► │ micro → features.js →   │
  │ "euh" "ah non" bruit  │  40 × 98    │  (avg ⊕ max pool) │          │ model.onnx → décodage   │
  └──────────────────────┘             └───────────────────┘          └────────────────────────┘
```

1. **Génération TTS** (`solfege/tts.py`) : chaque note est synthétisée sur une grille
   *voix × vitesse × hauteur*. Les **notes tenues** sont obtenues par **vocodeur de phase**
   (allonge la voyelle sans changer la hauteur) ; les **rapides** par vitesse espeak élevée.
   La classe *bruit* couvre mots parasites, quasi-homophones, silence et bruit rose/blanc.
2. **Features** (`solfege/features.py`) : log-mel normalisé. Réimplémenté à l'identique en
   JS (`web/features.js`) — la parité est **testée** (sinon le modèle se tromperait en ligne).
3. **Modèle** (`solfege/model.py`) : petit CNN avec **pooling global moyenne + maximum**.
   Le *max* garde la trace de l'attaque consonantique (le « d » de *do*, le « s » de *sol*)
   qui distingue deux voyelles tenues identiques — sans lui, on confond do↔sol et mi↔si.
4. **Décodage en flux** (`solfege/infer.py`, `web/app.js`) : fenêtre glissante, fusion des
   répétitions, **suppression du bruit** → la transcription ne contient que des notes.

---

## Résultats

Reproduits par `python -m solfege.train` puis `pytest` ; chiffres exacts dans
[`artifacts/metrics.json`](artifacts/metrics.json). Tout est mesuré sur des **rendus TTS
inédits** (combinaisons voix × vitesse × hauteur disjointes de l'entraînement).

| Mesure | Résultat |
|---|---|
| **Reconnaissance des notes** (bonne note quand c'en est une) — *cible gatée* | **99,7 %** ✓ |
| Rappel par note isolée (do…si) | **8/8 par note** |
| Rejet du bruit (parasites → classe *bruit*) | ~93 % |
| Accuracy globale 8 classes (notes + bruit) | ~98,9 % |
| Robustesse forte augmentation (bruit SNR ~8) | ~89 % |
| Voix jamais vues (timbres tenus à l'écart) | ~83 % |

La **cible « 99 % » porte sur la reconnaissance des notes** (« reconnaître do ré mi fa sol
la si »), gatée par `pytest`. Le rejet du non-solfège est une capacité **secondaire**
(intrinsèquement plus dure : rejeter un mot quelconque) reportée honnêtement — et améliorée
en pratique par le seuil de confiance du décodeur en flux.

Ce que `pytest tests/test_accuracy.py` vérifie, en re-synthétisant à chaque fois :
- **reconnaissance des notes ≥ 99 %** (rendus disjoints de l'entraînement) ;
- **rappel par note** (chaque note isolée bien reconnue) ;
- **parasites rejetés** (« euh », « ah non », bonjour, silence → *bruit*) ;
- **notes tenues** (« dooooo ») correctement classées ;
- **gamme décodée dans l'ordre** par le décodeur en flux (LCS ≥ 6/7).

---

## Démo en ligne (`.pages.dev`)

Le dossier `web/` est un site **statique** (aucun build). Déploiement Cloudflare Pages :

```bash
npx wrangler pages deploy web --project-name solfege-stt
# → https://solfege-stt.pages.dev
```

En CI, `.github/workflows/deploy.yml` publie à chaque push (secrets `CLOUDFLARE_API_TOKEN`
et `CLOUDFLARE_ACCOUNT_ID` à définir dans *Settings → Secrets and variables → Actions*).
Un **fallback GitHub Pages** (`pages.yml`, déclenchement manuel) permet une démo en ligne
même sans compte Cloudflare.

La page permet d'**écouter le micro** (chantez la gamme) et de **tester des extraits**
synthétisés (gamme lente/rapide, notes tenues, parasites) sans micro.

---

## Honnêteté / limites

- La précision est mesurée sur de la **voix synthétique** (espeak-ng). Sur de **vraies
  voix humaines**, chant réel, accents et micros variés, elle sera plus basse : ce POC
  démontre la **faisabilité et la chaîne complète**, pas une performance terrain.
- On reconnaît le **nom** de la note (la syllabe), **pas la hauteur** (fréquence chantée) :
  chanter « do » grave ou aigu donne la même classe `do`.
- Pour viser la robustesse sur voix réelles : collecter un petit corpus humain et
  **fine-tuner** (le même pipeline s'y prête), ou repartir d'un ASR pré-entraîné
  (Whisper) + **LoRA** — voir `agent/03-decisions.md` (D2) pour le compromis retenu ici.

---

## Structure

Voir [`agent/04-pipeline.md`](agent/04-pipeline.md) pour l'arborescence complète et les
commandes, et [`agent/`](agent/) pour la spécification *spec-first* (vision, besoins, spec,
décisions, pipeline).

## Licence

POC — usage libre. espeak-ng est sous GPLv3.
