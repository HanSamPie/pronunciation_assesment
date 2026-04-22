.PHONY: setup test train eval clean

# Python executable
PYTHON = python

setup:
	pip install -r requirements.txt
	pip install -e ".[dev]"

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
