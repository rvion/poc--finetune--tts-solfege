# Journal des décisions (ADR léger)

Format : décision · raison · alternatives écartées.

## D1 — Keyword spotting plutôt qu'ASR généraliste
7 syllabes + 1 classe bruit = petit vocabulaire fermé. Un CNN sur log-mel suffit,
tourne dans le navigateur et atteint la cible. *Écarté :* fine-tuner Whisper (lourd,
GPU, ~40–75 M params difficiles à faire tourner en temps réel côté client).

## D2 — Pas de LoRA / fine-tune d'un gros modèle (question B8)
L'utilisateur demandait « peut-être un fine-tune ? un LoRA ? ». Un LoRA s'applique à un
**grand** modèle pré-entraîné ; ici entraîner **de zéro** un petit CNN est plus simple,
100 % reproductible, léger à déployer et dépasse déjà 99 %. Le chemin « fine-tune Whisper
+ LoRA sur voix réelles » est documenté comme évolution (05/README) si on vise la robustesse
sur vraies voix humaines. *Écarté :* LoRA maintenant (complexité et déploiement navigateur
sans bénéfice mesurable sur ce vocabulaire).

## D3 — espeak-ng comme générateur TTS
Léger, déterministe, multi-voix/multi-langues, installable partout (apt). Sert à la fois à
fabriquer l'entraînement et à **tester** (jeu d'éval re-synthétisé). *Écarté :* Coqui/Piper
(modèles lourds à télécharger, moins de contrôle vitesse/hauteur simple).

## D4 — `config.json` source unique + parité Python/JS testée
Le modèle est entraîné en Python mais infère en JS : la moindre divergence des features
casse tout. Un seul fichier de config, deux implémentations réconciliées par des vecteurs
de référence (tolérance 1e-4). *Écarté :* recalculer les features côté serveur (pas de
serveur), ou faire confiance sans test (fragile).

## D5 — Pooling global moyenne ⊕ maximum
Premiers essais avec `AdaptiveAvgPool` : ~50 %, grosses confusions mi↔si et do↔sol. Cause :
la moyenne **dilue l'onset consonantique** (bref) qui distingue des voyelles tenues
identiques. Ajouter le **max** (présence d'un pic d'onset n'importe où) a débloqué la montée
en accuracy. *Écarté :* GRU/attention (plus lourd à exporter/faire tourner en JS).

## D6 — Onset conservé dans la fenêtre pour les notes
Une note tenue recadrée au milieu (« ooo » sans attaque) est intrinsèquement ambiguë
(do vs sol). `place_in_window(keep_onset=True)` garantit l'attaque → supprime ce bruit de
label. Réaliste : dans « ddooo…rémi », chaque note commence par sa consonne.

## D7 — Cible 99 % sur notes articulées ; robustesse reportée à part
Sur forte augmentation (bruit SNR ~8, étirements extrêmes) un petit modèle plafonne (~90 %).
La cible utilisateur porte sur des notes **délibérément lues** ; on gate donc à 99 % sur le
régime « clean » (notes articulées, rendus inédits) et on **reporte** séparément l'accuracy
en forte augmentation et sur voix jamais vues. Honnête et aligné sur l'usage. *Écarté :*
gater sur le pire cas (échouerait) ou masquer la robustesse (malhonnête).

## D8 — Déploiement Cloudflare Pages (Wrangler) + fallback GitHub Pages
Demande explicite : `.pages.dev` via Wrangler. `wrangler.toml` + workflow
`cloudflare/wrangler-action` (secrets `CLOUDFLARE_API_TOKEN`/`ACCOUNT_ID`). Le dépôt de
référence `screenplay` déployait en fait sur **GitHub Pages** ; on garde ce mécanisme en
**filet de sécurité** pour une démo en ligne même sans compte Cloudflare.

## D9 — Modèle ONNX committé dans `web/`
La démo doit marcher sans ré-entraîner et la CI doit vérifier l'artefact réellement livré.
`web/model.onnx` (~0,7 Mo) est donc versionné ; `pytest` le charge et le teste sur du TTS frais.

## D10 — Features à 2 canaux : log-mel ⊕ Δ (dérivée temporelle)
Sans les deltas, les confusions résiduelles se concentraient sur des paires ne se distinguant
que par l'**onset consonantique** (mi↔si = même voyelle « i », fa↔la = même « a »). Ajouter le
canal Δ (transition temporelle) donne explicitement au modèle l'information d'attaque, ce qui a
resserré ces confusions. Δ est une opération linéaire simple, **répliquée à l'identique en JS**
(parité testée). *Écarté :* MFCC (moins lisible, parité plus fragile), features main-only.

## D11 — Page autonome (inférence CNN en JS pur) en complément d'onnxruntime
En plus de la démo `web/` (onnxruntime-web, pour un vrai hébergement), un script
(`scripts/export_standalone.py` + `web/nn.js`) génère une **page HTML unique auto-contenue** :
poids du CNN embarqués en base64, forward réimplémenté en JavaScript pur (miroir de `model.py`),
**aucune dépendance externe**. Utile pour héberger la démo n'importe où sans CDN. La correction
du forward JS pur est **vérifiée contre PyTorch** (`scripts/verify_standalone.mjs`, écart < 2e-3).
*Contexte :* onnxruntime-web reste la voie principale (plus rapide, GPU/WASM) ; le JS pur est un
filet « zéro dépendance ».
