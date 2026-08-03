# Automated Pronunciation Assessment (APA)

A hierarchical multi-task deep learning system for automated pronunciation assessment, trained on **Speechocean762**. Predicts pronunciation quality at phoneme, word, and sentence level using a Bi-directional GRU with attention pooling. Compares against Linear Regression and Decision Tree / XGBoost baselines.

All experiments are fully reproducible via fixed random seeds, Hydra configuration management, and MLflow experiment tracking.

---

## Table of Contents

1. [Cookbook Quick Reference](#cookbook-quick-reference)
2. [Requirements](#1-requirements)
3. [Installation](#2-installation)
4. [Dataset Setup](#3-dataset-setup)
5. [Configuration](#4-configuration)
6. [Running the Pipeline](#5-running-the-pipeline)
   - [Step 1 — Data Splitting](#step-1--data-splitting)
   - [Step 2 — Forced Alignment](#step-2--forced-alignment)
   - [Step 3 — Feature Extraction](#step-3--feature-extraction)
   - [Step 4 — Normalization](#step-4--normalization)
   - [Step 5 — Training](#step-5--training)
   - [Step 6 — Evaluation](#step-6--evaluation)
7. [Baselines](#6-baselines)
8. [Visualizations](#7-visualizations)
9. [MLflow Tracking](#8-mlflow-tracking)
10. [Reproducibility Notes](#9-reproducibility-notes)
11. [Project Structure](#10-project-structure)

---

## Cookbook Quick Reference

Here are common copy-paste recipes for running, configuring, and evaluating the project:

### Recipe 1: End-to-End Pipeline in Two Commands
```bash
make prep   # Runs split, alignment, feature extraction, & normalization automatically
make train  # Trains all models (BiGRU, Linear Baseline, Tree Baseline)
make eval   # Evaluates all models & generates charts
```

### Recipe 2: Training a Specific Model
```bash
make train MODEL=bigru    # Train BiGRU primary model
make train MODEL=linear   # Train Linear Regression baseline
make train MODEL=tree     # Train Decision Tree / XGBoost baseline
```

### Recipe 3: Hyperparameter & Loss Weight Overrides
```bash
# Override loss weights dynamically
python -m src.training.trainer +model=bigru loss_weights.phoneme=1.0 loss_weights.word=3.0 loss_weights.sentence=5.0

# Toggle between major scores (accuracy only) and all metrics
python -m src.training.trainer +model=bigru score_mode=all_metrics
```

### Recipe 4: MLflow Experiment Tracking UI
```bash
make mlflow-ui
# Open http://localhost:5000 in your browser
```

---

## 1. Requirements

- Python ≥ 3.10
- [Montreal Forced Aligner (MFA)](https://montreal-forced-aligner.readthedocs.io/) installed and on `PATH`
- [openSMILE](https://audeering.github.io/opensmile-python/) (installed via pip)
- CUDA-capable GPU recommended for Bi-GRU training (CPU is supported but slow)

---

## 2. Installation

```bash
# Clone the repository
git clone <repo-url>
cd pronunciation_assesment

# Create and activate a virtual environment
python -m venv .venv
source .venv/bin/activate

# Install Python dependencies
pip install -r requirements.txt

# Download the MFA english_mfa acoustic model
mfa model download acoustic english_mfa
mfa model download dictionary english_mfa
```

---

## 3. Dataset Setup

1. Download **Speechocean762** from [OpenSLR](https://www.openslr.org/101/).
2. Extract the archive into `data/raw/`:

```
data/raw/
├── wav/           # Audio files (.wav)
├── text           # Transcripts
├── resource/      # Annotation files (scores, speaker metadata)
└── ...
```

3. Verify that demographic annotations (child / adult labels) are present in the metadata files — these are required for fairness analysis.

---

## 4. Configuration

All hyperparameters, paths, and flags are managed via **Hydra** YAML files in `configs/`. You should **never hardcode values** in scripts.

### Key file: `configs/base.yaml`

```yaml
# Reproducibility
seed: 42

# Data paths
data_dir: data/raw
splits_dir: data/splits
features_dir: data/features
scalers_dir: data/scalers

# Forced alignment
mfa_model: english_mfa

# Feature extraction
feature_store: hdf5          # Storage format (HDF5)
use_boundary_jitter: false   # Set true to enable ±5ms boundary jitter

# Loss weights (easily editable)
loss_weights:
  phoneme: 1.0
  word:    2.0
  sentence: 5.0
```

To override any value at runtime without editing the file, use Hydra's override syntax:

```bash
# Example: enable boundary jitter for a specific run
python -m src.training.trainer use_boundary_jitter=true
```

---

## 5. Running the Pipeline

Run each step in order. Alternatively, run `make prep` to execute steps 1–4 automatically (using dataset alignments). Steps 5–6 can be repeated for different model configurations.

### Step 1 — Data Splitting

Splits speakers into Train (70%) / Validation (15%) / Test (15%). **No speaker appears in more than one partition.**

```bash
make split
```

Output: manifest files written to `data/splits/` (train.csv, val.csv, test.csv).

---

### Step 2 — Forced Alignment

Extracts phoneme-level timing boundaries ($t_{start}$, $t_{end}$). You can either use the pre-computed alignments shipped with the dataset or run MFA.

```bash
# Use pre-computed dataset alignments (Recommended)
make align-dataset

# OR run full MFA alignment
make align-mfa
```

> **Note:** MFA requires the `english_mfa` acoustic model (downloaded in the Installation step). Alignment may take 30–60 minutes on the full dataset. TextGrid output is cached.

---

### Step 3 — Feature Extraction

Extracts **eGeMAPS LLD's (25 features)** from the audio file using openSMILE. The frames are then assigned to the corresponding phonemes based on the TextGrid alignment. Features are serialized to HDF5.

```bash
make extract
```

To enable boundary jittering (±5 ms) for a training run:

```bash
python -m src.data.extract use_boundary_jitter=true
```

Output: `data/features/features.h5`, keyed by `speaker_id / sentence_id / phoneme_idx`.

---

### Step 4 — Normalization

Fits a `StandardScaler` on the **training partition only**, then applies it to validation and test. The scaler object is saved for use during evaluation.

```bash
make normalize
```

Output: `data/scalers/scaler.joblib`.

> **Important:** Never re-fit the scaler on validation or test data. The saved scaler is also registered as an MLflow artifact on each training run.

---

### Step 5 — Training

Trains the models. By default, running `make train` will execute training for all models (BiGRU, Linear, Tree) sequentially. You can train a specific model by passing `MODEL=<name>`. All metrics, git commit hash, and configs are logged to MLflow.

```bash
# Train all models
make train

# Train only Bi-GRU
make train MODEL=bigru
```

To override hyperparameters for a specific experiment, you can still run the trainer module directly:
```bash
python -m src.training.trainer +model=bigru loss_weights.word=3.0
```

#### Bi-GRU Model Architecture (`configs/model/bigru.yaml`)
The primary model uses hierarchical multi-task learning. Its default architecture includes:
* **Input**: 25 eGeMAPS LLD's as features.
* **Encoder**: 3-layer Bi-GRU, hidden size 32 (64 total).
* **Regularization**: Dropout 0.3, L2 weight decay 1e-4.
* **Training**: Adam optimizer, initial LR 0.0005, batch size 64.
  * **Learning Rate Scheduler**: `ReduceLROnPlateau` dynamically reduces the LR by a factor of 0.5 if validation loss fails to improve for 3 epochs (lower bound 1e-6).
  * **Early Stopping**: Halts training if validation loss plateaus (minimum change 0.0005) for 8 consecutive epochs, and automatically restores the best model weights.
* **Pooling & Prediction**: Learned scalar attention pooling at word and sentence levels. Prediction heads use 2-layer MLPs with ReLU activation, 20% dropout, and scaled Sigmoid outputs.

The best validation checkpoints are saved as MLflow artifacts.

---

### Step 6 — Evaluation

Runs deterministic multi-split evaluation (train, val, test) across all models. Results are cached in a local **SQLite database** (`results.db`) by MD5 hash. Re-running with identical inputs returns the cached result instantly.

```bash
make eval
```

Evaluation outputs are organized into model-specific directories: `outputs/evaluation/{split}/{model_type}/`. Dropout and boundary jitter are explicitly disabled during evaluation to ensure determinism.

To clear the cache and force a re-evaluation:
```bash
make eval-clean
```

---

## 6. Baselines

The pipeline includes two classical baselines that operate on statically pooled features. Both are automatically trained when running `make train`.

* **Linear Regression** (`linear`): Single fit, no epochs and batch size of 256.
* **Decision Tree / XGBoost** (`tree`): Ensemble with max depth 4, min samples per leaf 5.

Their settings can be found in `configs/model/linear.yaml` and `configs/model/tree.yaml`.

---

## 7. Visualizations

Publication-ready charts are generated automatically at the end of evaluation and organized in `outputs/evaluation/`. The visualization pipeline includes standardized theme styles and model color coding for consistency.

Charts generated per split:
* Predicted vs. human score scatter plots
* Correlation matrix heatmaps
* Error distribution histograms
* Model comparison bar charts across splits
* Demographic bias / fairness analysis (Children vs. Adults)
* Loss curves and score accuracy analysis comparisons

You can also run independent analyses:
```bash
# Analyze dataset score distributions
make analyze

# Attention weight heatmap for a selected sentence
# python -m src.visualization.attention --run-id <mlflow-run-id>
```

---

## 8. MLflow Tracking

Start the MLflow UI to inspect all runs:

```bash
mlflow ui --port 5000
```

Then open [http://localhost:5000](http://localhost:5000) in your browser.

Each run automatically logs:

| Logged Item | Type |
|---|---|
| Git commit hash | Tag |
| Full Hydra config (YAML) | Artifact |
| Global random seeds | Parameter |
| Loss weights | Parameter |
| Train / val loss per epoch | Metric |
| Final model weights | Artifact |
| Fitted `StandardScaler` | Artifact |
| All 4 visualization charts | Artifact |
| Evaluation report (PCC, RMSE, SRC) | Artifact |

---

## 9. Reproducibility Notes

- **Seeds**: Python, NumPy, and PyTorch seeds are all set from `configs/base.yaml` (`seed: 42`) and logged to MLflow.
- **Scaler**: Fit once on training data; the `scaler.joblib` artifact ensures identical normalization on every evaluation run.
- **Evaluation cache**: The SQLite cache in `results.db` uses an MD5 hash of (config + model weights + test dataset) — identical inputs always return identical, cached outputs.
- **Git tagging**: Tag the commit corresponding to your final experiment run for long-term traceability:
  ```bash
  git tag -a experiment-final -m "Final experiment run"
  git push origin experiment-final
  ```

---

## 10. Project Structure

```
pronunciation_assesment/
├── configs/                    # Hydra YAML configurations
│   ├── base.yaml               # Global defaults (seed, paths, loss weights)
│   ├── model/
│   │   ├── bigru.yaml
│   │   ├── linear.yaml
│   │   └── tree.yaml
│   └── experiment/
│       └── default.yaml
├── data/
│   ├── raw/                    # Speechocean762 dataset (not tracked by git)
│   ├── splits/                 # Speaker-independent split manifests
│   ├── features/               # Serialized eGeMAPS features (HDF5)
│   └── scalers/                # Fitted StandardScaler objects
├── src/
│   ├── data/
│   │   ├── split.py            # Speaker-independent splitting
│   │   ├── align.py            # MFA forced alignment wrapper
│   │   ├── extract.py          # openSMILE eGeMAPS extraction
│   │   ├── persist.py          # HDF5 serialization
│   │   └── normalize.py        # StandardScaler fit/transform
│   ├── models/
│   │   ├── bigru.py            # Hierarchical Multi-Task Bi-GRU
│   │   ├── linear_baseline.py  # Linear Regression baseline
│   │   └── tree_baseline.py    # Decision Tree / XGBoost baseline
│   ├── training/
│   │   ├── trainer.py          # Training loop + MLflow logging
│   │   └── loss.py             # Weighted multi-task loss
│   ├── evaluation/
│   │   ├── evaluate.py         # Deterministic evaluation (PCC, RMSE, SRC)
│   │   ├── cache.py            # SQLite MD5 result cache
│   │   └── fairness.py         # Stratified fairness analysis
│   └── visualization/
│       ├── scatter.py          # Scatter plots
│       ├── loss_curves.py      # Loss curves
│       ├── attention.py        # Attention heatmaps
│       └── fairness_charts.py  # Fairness bar charts
├── notebooks/                  # Exploratory analysis notebooks
├── mlruns/                     # MLflow tracking (not tracked by git)
├── results.db                  # SQLite evaluation cache
├── requirements.txt
├── APA_Antigravity.md          # Original project specification
└── README.md                   # This file
```
