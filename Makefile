# PYTHON переопределяется под целевую машину: make venv PYTHON=python3.11
PYTHON ?= python3
VENV   := .venv
PY     := $(VENV)/bin/python

.PHONY: venv venv-gpu check models manifest qc splits data labeling test clean-cache

venv:
	$(PYTHON) -m venv $(VENV) && $(PY) -m pip install -q -U pip && $(PY) -m pip install -q -r requirements.txt

# Зависимости GPU-этапа. torch ставится отдельно — см. README.
venv-gpu:
	$(PY) -m pip install -r requirements-gpu.txt

# Диагностика машины: запускать первой на новой машине, вывод присылать целиком.
check:
	$(PY) scripts/check_env.py

# Предзагрузка весов: сетевые проблемы вылезают за минуту, а не через час.
models:
	$(PY) scripts/fetch_models.py

manifest:
	$(PY) scripts/build_manifest.py

qc:
	$(PY) scripts/run_qc.py

splits:
	$(PY) scripts/make_splits.py

# Полный CPU-конвейер Ф0: ~6 минут на 78k кадров.
data: manifest qc splits

# Калибровочные наборы для ручной разметки -> cache/labeling/*.html
labeling:
	$(PY) scripts/make_labeling_set.py

# GPU-этап целью не оформлен намеренно: стадиям нужны флаги (--shard,
# --batch-size, --limit), и при отладке важно видеть, какая именно команда
# упала. Запускается напрямую, см. README.

test:
	$(PY) -m pytest tests/ -q

clean-cache:
	rm -rf cache
