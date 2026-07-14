# Besoins (exprimés par l'utilisateur)

Source : brief vocal (2026-07).

## Fonctionnels
- **B1 — Récupérer un petit modèle TTS.** Utilisé comme *générateur de données*
  (espeak-ng : léger, déterministe, multi-voix).
- **B2 — Reconnaître les notes** do ré mi fa sol la si (do), en entrée parlée/chantée.
- **B3 — Deux régimes de lecture** : très **rapide** (« dorémifasol » enchaîné) et
  **tenu** (« ddooooooooorémiiiiiré »). Les deux doivent être reconnus.
- **B4 — Rejeter le non-solfège.** Les mots hors solfège (« euh… », « ah non… »),
  hésitations, silences et bruits sont transcrits comme **bruit inintéressant** et
  écartés : on ne garde que les notes.
- **B5 — ≥ 99 % d'accuracy**, vérifiable *dans le dépôt* (tout le nécessaire présent).
- **B6 — Démonstrateur en ligne.** Un site public qui fait la démonstration ;
  déploiement via **Wrangler sur `.pages.dev`** (méthode vue dans le dépôt `screenplay`).
- **B7 — Tests critiques**, incluant l'usage du **TTS pour tester le STT**.
- **B8 — Fine-tune / LoRA ?** Question ouverte de l'utilisateur (cf. 03-decisions D2).

## Hypothèses
- **H1** Un vocabulaire de 7 syllabes + 1 classe « bruit » se traite comme du
  *keyword spotting* : un classifieur audio compact suffit (pas besoin d'un gros ASR).
- **H2** La cible « 99 % » porte sur des notes **délibérément articulées** (cas d'usage :
  quelqu'un chante/annonce des notes). La robustesse au bruit fort est un objectif
  secondaire, mesuré séparément.
- **H3** Un modèle entraîné sur du TTS multi-voix généralise raisonnablement ; la
  validation se fait sur des rendus TTS inédits et des voix tenues à l'écart.

## Non-objectifs
- Détection de **hauteur** (fréquence chantée) : ici on reconnaît le **nom** de la note
  (la syllabe), pas la fréquence fondamentale émise.
- Un ASR généraliste ou un grand modèle : hors périmètre (léger + navigateur).
- Garantie de 99 % sur **vraies voix humaines** : non prouvé ici (données synthétiques).
  Le chemin pour y arriver (collecte/fine-tune sur voix réelles) est documenté.
