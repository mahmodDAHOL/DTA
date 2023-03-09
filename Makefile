VENV = venv
PYTHON = ./venv/bin/python
PIP = ./venv/bin/pip
RUFF = ./venv/bin/ruff
BLACK = ./venv/bin/black

all: install format

install: requirements.txt
	apt install python3.9-venv
	python -m venv $(VENV)
	$(PYTHON) -m pip install --upgrade pip
	$(PIP) install -r requirements.txt
	$(PIP) install --no-index torch-scatter -f https://pytorch-geometric.com/whl/torch-1.12.1+cpu.html
	$(PIP) install --no-index torch-sparse -f https://pytorch-geometric.com/whl/torch-1.12.1+cpu.html
	$(PIP) install --no-index torch-cluster -f https://pytorch-geometric.com/whl/torch-1.12.1+cpu.html
	$(PIP) install --no-index torch-spline-conv -f https://pytorch-geometric.com/whl/torch-1.12.1+cpu.html
	$(PIP) install torch-geometric

format: $(VENV)/bin/ruff
	$(RUFF) check --fix --select ALL --ignore "E712","COM812","D212","D203" .
	$(BLACK) .

run: $(VENV)/bin/python
	$(PYTHON) data_process.py --fold-number 4
	$(PYTHON) training_5folds.py --cuda-name "" --fold-number 4 
	$(PYTHON) test.py --dataset-name "davis" --cuda-name "" --fold-number 4

clean:
	rm -rf __pycache__ .ruff_cache
	rm -rf $(VENV)