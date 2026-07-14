# Vision

## Pourquoi ce dépôt
Prouver qu'on peut, **sans enregistrer une seule vraie voix**, obtenir un
**reconnaisseur de notes de solfège** fiable et **déployable en ligne** :
1. un **modèle de synthèse vocale** (TTS, espeak-ng) fabrique le corpus ;
2. un **classifieur / STT compact** apprend à transcrire **do ré mi fa sol la si** ;
3. tout ce qui n'est pas une note (« euh… », « ah non… », silence, bruit) est
   étiqueté **bruit** et jeté, pour ne garder que la suite de notes ;
4. une **démo web** (Cloudflare Pages / `.pages.dev`) fait tourner le modèle
   **dans le navigateur** en temps réel via le micro.

## Principes
1. **TTS pour fabriquer ET pour tester.** Le même moteur génère l'entraînement et un
   jeu d'évaluation frais : « utiliser du text-to-speech pour tester le speech-to-text ».
2. **Source unique de vérité.** `config.json` fixe SR, STFT, mel et labels ; Python et
   JS le lisent — aucun paramètre en double.
3. **Parité stricte Python ↔ navigateur.** Les features log-mel sont réimplémentées à
   l'identique en JS (vérifié à 1e-4), sinon le modèle entraîné en Python se trompe en ligne.
4. **Léger et déployable.** CNN ~150k paramètres exporté en ONNX ; inférence temps réel
   via onnxruntime-web. Aucune infrastructure serveur.
5. **Robuste aux tempos.** Notes **tenues** (étirement temporel/vocodeur de phase) et
   **rapides** (vitesse espeak élevée) couvertes par l'augmentation.
6. **Honnête sur les limites.** L'accuracy est mesurée sur de la voix **synthétique** ;
   voir 01-besoins pour ce que cela implique sur de vraies voix humaines.

## Définition de « terminé »
- `python -m solfege.train` reproduit un modèle qui dépasse **99 %** sur des notes
  articulées TTS inédites ; `pytest` le vérifie automatiquement.
- Le modèle (`web/model.onnx`) est committé et la **démo web** reconnaît les notes au
  micro + sur des extraits fournis, en ignorant le bruit.
- Le déploiement `.pages.dev` (Wrangler) est configuré ; un fallback GitHub Pages existe.
- `agent/` documente vision, besoins, spec, décisions et pipeline.
