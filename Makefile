VENV = venv
PYTHON = ./venv/bin/python
PIP = ./venv/bin/pip
RUFF = ./venv/bin/ruff
BLACK = ./venv/bin/black
GDOWN = ./venv/bin/gdown
IGNORE = "E712","COM812","D212","D203","UP007","N806","TCH003","TCH002","TCH001","D105","ANN101","PLR0913","PGH003","D102","D107","D205","FBT001","FBT002","B905","B008","G004","FBT003"
DATASET_LINK = https://drive.google.com/file/d/1CJIvzSDgZXSgTB5CpCutwShs7Xs2rEkk/view?usp=share_link

all: install run

install: requirements.txt
	pip install virtualenv
	virtualenv -p python3.10 $(VENV)
	$(PYTHON) -m pip install --upgrade pip
	$(PIP) install -r requirements.txt
	$(GDOWN) --fuzzy ${DATASET_LINK}
	unzip davis.zip 
	rm davis.zip

format: $(VENV)/bin/ruff
	$(RUFF) check --fix --select ALL --ignore  ${IGNORE} src
	$(BLACK) src

test: tests
	python -m tests.generate_small_data
	python -m pytest 

run: $(VENV)/bin/python
	$(PYTHON) -m src.training_5folds --cuda-name "cuda:0" --dataset-name "davis" --fold-number 1
	$(PYTHON) -m src.training_5folds --cuda-name "cuda:0" --dataset-name "davis" --fold-number 2
	$(PYTHON) -m src.training_5folds --cuda-name "cuda:0" --dataset-name "davis" --fold-number 3
	$(PYTHON) -m src.training_5folds --cuda-name "cuda:0" --dataset-name "davis" --fold-number 4
	$(PYTHON) -m src.training_5folds --cuda-name "cuda:0" --dataset-name "davis" --fold-number 5
	$(PYTHON) -m src.eval --dataset-name "davis" --cuda-name "cuda:0" --fold-number 1
	$(PYTHON) -m src.eval --dataset-name "davis" --cuda-name "cuda:0" --fold-number 2
	$(PYTHON) -m src.eval --dataset-name "davis" --cuda-name "cuda:0" --fold-number 3
	$(PYTHON) -m src.eval --dataset-name "davis" --cuda-name "cuda:0" --fold-number 4
	$(PYTHON) -m src.eval --dataset-name "davis" --cuda-name "cuda:0" --fold-number 5

clean:
	rm -rf __pycache__ .ruff_cache .pytest_cache
	rm -rf $(VENV)
	rm -rf data