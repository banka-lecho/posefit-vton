VENV := .venv
PY   := $(VENV)/bin/python

.PHONY: venv manifest qc splits data labeling test clean-cache

venv:
	python3.12 -m venv $(VENV) && $(PY) -m pip install -q -U pip && $(PY) -m pip install -q -r requirements.txt

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

test:
	$(PY) -m pytest tests/ -q

clean-cache:
	rm -rf cache
