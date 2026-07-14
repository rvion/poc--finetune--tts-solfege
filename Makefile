.PHONY: setup train test fixtures samples serve deploy clean

setup:          ## installe espeak-ng + dépendances Python
	sudo apt-get install -y --no-install-recommends espeak-ng
	pip install -r requirements.txt

train:          ## entraîne le modèle et exporte web/model.onnx
	python -m solfege.train --n-per-class 1800 --epochs 30

fixtures:       ## régénère les vecteurs de parité Python<->JS
	python scripts/make_fixtures.py

samples:        ## régénère les extraits .wav de démo
	python scripts/gen_samples.py

test:           ## lance tous les tests (features, TTS, accuracy >= 99%)
	pytest

serve:          ## prévisualise la démo sur http://localhost:8000
	python -m http.server -d web 8000

deploy:         ## déploie web/ sur Cloudflare Pages (.pages.dev)
	npx wrangler pages deploy web --project-name solfege-stt

clean:
	rm -rf artifacts_smoke artifacts_test __pycache__ */__pycache__ .pytest_cache
