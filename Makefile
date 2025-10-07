# ==== GENESIS-AI Makefile ====

.PHONY: setup test app format lint typecheck

# Set default python
PYTHON = python3

setup:
	$(PYTHON) -m pip install --upgrade pip
	$(PYTHON) -m pip install -e .[dev]

test:
	$(PYTHON) -m pytest -v --maxfail=1 --disable-warnings -q

app:
	streamlit run src/genesis_ai/app/main.py

format:
	black src tests
	isort src tests

lint:
	ruff check src tests

typecheck:
	mypy src

# === Autonomous System Targets ===
init-db:
	$(PYTHON) -m genesis_ai.db.models

serve:
	$(PYTHON) -m uvicorn genesis_ai.inference.service:app --host 0.0.0.0 --port 8000

scheduler:
	$(PYTHON) -m genesis_ai.training.scheduler

monitor:
	$(PYTHON) -m genesis_ai.monitor.alerts

# === Convenience targets for Colab ===
colab-setup:
	pip install -e .[dev]

colab-app:
	!streamlit run src/genesis_ai/app/main.py --server.port=8080
