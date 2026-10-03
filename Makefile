PYTHON ?= python
UV ?= uv

setup:
	$(UV) sync --extra asr --extra benchmark --extra database
	cp -n .env.example .env || true

run:
	$(UV) run uvicorn app.main:app --reload

test:
	$(UV) run pytest -m "not integration" -q

benchmark:
	$(UV) run python scripts/run_benchmark.py --limit 5 --model base

smoke:
	$(UV) run python scripts/smoke_test.py

verify:
	$(UV) run python scripts/verify_setup.py
