VENV = venv
PYTHON = ./venv/bin/python
PIP = ./venv/bin/pip
RUFF = ./venv/bin/ruff
BLACK = ./venv/bin/black
IGNORE = "E712","COM812","D212","D203","UP007","N806","TCH003","TCH002","TCH001","D105","ANN101","PLR0913","PGH003","D102","D107","D205","FBT001","FBT002","B905","B008","G004"

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
	$(RUFF) check --fix --select ALL --ignore  ${IGNORE} .
	$(BLACK) .

run: $(VENV)/bin/python
	$(PYTHON) training_5folds.py --cuda-name "" 
	$(PYTHON) test.py --dataset-name "davis" --cuda-name "" --fold-number 4

clean:
	rm -rf __pycache__ .ruff_cache
	rm -rf $(VENV)