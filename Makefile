.PHONY: setup test split align-mfa align-dataset extract normalize prep train mlflow-ui eval clean

# Python executable
PYTHON = python

setup:
	pip install -r requirements.txt
	pip install -e ".[dev]"

split:
	$(PYTHON) -m src.data.split

align-mfa:
	$(PYTHON) -m src.data.align

align-dataset:
	$(PYTHON) -m src.data.align_dataset

extract:
	$(PYTHON) -m src.data.extract

normalize:
	$(PYTHON) -m src.data.normalize

prep: split align-dataset extract normalize

test:
	pytest

# Model to train: bigru | linear | tree | all (default: all)
MODEL ?= all

ifeq ($(MODEL),all)
	# No +model override — trainer detects missing cfg.model.name and runs all models
TRAIN_CMD = $(PYTHON) -m src.training.trainer
else
TRAIN_CMD = $(PYTHON) -m src.training.trainer +model=$(MODEL)
endif

train:
	$(TRAIN_CMD)

mlflow-ui:
	$(PYTHON) -m mlflow server --host 127.0.0.1 --port 5000

eval:
	$(PYTHON) -m src.evaluation.evaluate

clean:
	find . -type d -name "__pycache__" -exec rm -r {} +
	find . -type d -name ".pytest_cache" -exec rm -r {} +
	rm -rf build/ dist/ *.egg-info/
