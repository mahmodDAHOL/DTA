VENV = venv
PYTHON = ./venv/bin/python
PIP = ./venv/bin/pip
RUFF = ./venv/bin/ruff
BLACK = ./venv/bin/black
GDOWN = ./venv/bin/gdown
IGNORE = "E712","COM812","D212","D203","UP007","N806","TCH003","TCH002","TCH001","D105","ANN101","PLR0913","PGH003","D102","D107","D205","FBT001","FBT002","B905","B008","G004","FBT003"
DATASET_LINK = https://drive.google.com/file/d/1CJIvzSDgZXSgTB5CpCutwShs7Xs2rEkk/view?usp=share_link

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
	$(GDOWN) --fuzzy ${DATASET_LINK}
	unzip davis.zip 
	rm davis.zip

format: $(VENV)/bin/ruff
	$(RUFF) check --fix --select ALL --ignore  ${IGNORE} .
	$(BLACK) .

run: $(VENV)/bin/python
	$(PYTHON) training_5folds.py --cuda-name "cuda:0" --dataset-name "davis" --fold-number 4
	$(PYTHON) test.py --dataset-name "davis" --cuda-name "cuda:0" 

clean:
	rm -rf __pycache__ .ruff_cache
	rm -rf $(VENV)
	rm -rf data