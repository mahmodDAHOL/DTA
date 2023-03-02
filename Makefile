all: install forrmat

install:
	python -m venv venv
	.\venv\Scripts\activate
	.\venv\Scripts\python.exe -m pip install --upgrade pip
	.\venv\Scripts\pip.exe install -r requirements.txt

forrmat:
	ruff check --fix --select ALL --ignore "E712","COM812" .

run:
	.\venv\Scripts\python.exe data_process.py --fold-number 4
	.\venv\Scripts\python.exe training_5folds.py --cuda_name "" fold-number 4 
	.\venv\Scripts\python.exe test.py --dataset-name "davis" --cuda_name "" fold-number 4
