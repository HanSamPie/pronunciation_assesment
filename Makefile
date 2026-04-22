.PHONY: setup test split align extract normalize prep train eval clean

# Python executable
PYTHON = python

setup:
	pip install -r requirements.txt
	pip install -e ".[dev]"

split:
	$(PYTHON) -m src.data.split

align:
	$(PYTHON) -m src.data.align

extract:
	$(PYTHON) -m src.data.extract

normalize:
	$(PYTHON) -m src.data.normalize

prep: split align extract normalize

test:
	pytest

train:
	$(PYTHON) -m src.training.trainer

eval:
	$(PYTHON) -m src.evaluation.evaluate

clean:
	find . -type d -name "__pycache__" -exec rm -r {} +
	find . -type d -name ".pytest_cache" -exec rm -r {} +
	rm -rf build/ dist/ *.egg-info/
