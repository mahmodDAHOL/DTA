VENV = venv
PYTHON = ./venv/bin/python
PIP = ./venv/bin/pip
RUFF = ./venv/bin/ruff

all: install format

install: requirements.txt
	python -m venv $(VENV)
	$(PYTHON) -m pip install --upgrade pip
	$(PIP) install -r requirements.txt

format: $(VENV)/bin/ruff
	$(RUFF) check --fix --select ALL --ignore "E712","COM812","D212","D203" .

run: $(VENV)/bin/python
	$(PYTHON) data_process.py --fold-number 4
	$(PYTHON) training_5folds.py --cuda_name "" fold-number 4 
	$(PYTHON) test.py --dataset-name "davis" --cuda_name "" fold-number 4

clean:
	rm -rf __pycache__ .ruff_cache
	rm -rf $(VENV)