# Automated Pronunciation Assessment (APA)

A hierarchical multi-task deep learning system for automated pronunciation assessment, trained on **Speechocean762**. Predicts pronunciation quality at phoneme, word, and sentence level using a Bi-directional GRU with attention pooling. Compares against Linear Regression and Decision Tree / XGBoost baselines.

All experiments are fully reproducible via fixed random seeds, Hydra configuration management, and MLflow experiment tracking.

---

## Table of Contents

1. [Requirements](#1-requirements)
2. [Installation](#2-installation)
3. [Dataset Setup](#3-dataset-setup)
4. [Configuration](#4-configuration)
5. [Running the Pipeline](#5-running-the-pipeline)
   - [Step 1 — Data Splitting](#step-1--data-splitting)
   - [Step 2 — Forced Alignment](#step-2--forced-alignment)
   - [Step 3 — Feature Extraction](#step-3--feature-extraction)
   - [Step 4 — Normalization](#step-4--normalization)
   - [Step 5 — Training](#step-5--training)
   - [Step 6 — Evaluation](#step-6--evaluation)
6. [Baselines](#6-baselines)
7. [Visualizations](#7-visualizations)
8. [MLflow Tracking](#8-mlflow-tracking)
9. [Reproducibility Notes](#9-reproducibility-notes)
10. [Project Structure](#10-project-structure)

---

## 1. Requirements

- Python ≥ 3.10
- [openSMILE](https://audeering.github.io/opensmile-python/) (installed via pip)
- CUDA-capable GPU recommended for Bi-GRU training (CPU is supported but slow)
- *Optional*: [Montreal Forced Aligner (MFA)](https://montreal-forced-aligner.readthedocs.io/) installed and on `PATH` (only needed if regenerating alignments instead of using the dataset defaults)

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

# (Optional) Download the MFA english_mfa acoustic model if using MFA alignments
# mfa model download acoustic english_mfa
# mfa model download dictionary english_mfa
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

# Forced alignment source
alignment_source: dataset    # "dataset" (pre-computed) or "mfa"

# Feature extraction
feature_store: hdf5          # Storage format (HDF5)
use_boundary_jitter: false   # Set true to enable ±5ms boundary jitter

# Loss weights (easily editable)
loss_weights:
  phoneme: 1.0
  word:    2.0
  sentence: 5.0

# Score Modeling Strategy
score_mode: major_scores  # Toggle "major_scores" or "all_metrics"
```

To override any value at runtime without editing the file, use Hydra's override syntax:

```bash
# Example: enable boundary jitter for a specific run
python -m src.training.trainer use_boundary_jitter=true
```

---

## 5. Running the Pipeline

Run each step in order. Steps 1–4 are one-time data preparation steps; steps 5–6 can be repeated for different model configurations.

### Step 1 — Data Splitting

Splits speakers into Train (70%) / Validation (15%) / Test (15%). **No speaker appears in more than one partition.**

```bash
python -m src.data.split
```

Output: manifest files written to `data/splits/` (train.csv, val.csv, test.csv).

---

### Step 2 — Alignment Verification

By default, the pipeline uses the pre-computed ground-truth word and phoneme boundary alignments provided with the Speechocean762 dataset. Alternatively, you can use the Montreal Forced Aligner (MFA) to generate boundaries.

```bash
# If using MFA (ensure mfa is installed and alignment_source: mfa is set in config)
python -m src.data.align
```

> **Note:** MFA requires the `english_mfa` acoustic model. Alignment may take 30–60 minutes on the full dataset. Using the pre-computed `dataset` alignments requires no extra steps.

---

### Step 3 — Feature Extraction

Extracts **eGeMAPS v02 Low-Level Descriptors (LLDs, 25 features per frame)** using openSMILE, natively extracting a continuous feature grid over the waveform, and sequentially aligning these frames strictly within phoneme boundaries. Features are serialized to HDF5.

```bash
python -m src.data.extract
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
python -m src.data.normalize
```

Output: `data/scalers/scaler.joblib`.

> **Important:** Never re-fit the scaler on validation or test data. The saved scaler is also registered as an MLflow artifact on each training run.

---

### Step 5 — Training

Trains the primary Hierarchical Multi-Task Bi-GRU model. All hyperparameters are read from config. All metrics, the git commit hash, and the config are automatically logged to MLflow.

```bash
python -m src.training.trainer
```

To override loss weights for a specific experiment:

```bash
python -m src.training.trainer \
  loss_weights.phoneme=1.0 \
  loss_weights.word=3.0 \
  loss_weights.sentence=8.0
```

The best validation checkpoint is saved as an MLflow artifact.

---

### Step 6 — Evaluation

Runs deterministic evaluation (PCC, RMSE, Spearman) on the test set. Results are cached in a local **SQLite database** (`results.db`) by MD5 hash. Re-running with identical inputs returns the cached result instantly.

```bash
python -m src.evaluation.evaluate --run-id <mlflow-run-id>
```

Fairness metrics (Children vs. Adults) are computed and reported automatically.

---

## 6. Baselines

Run the two classical baselines independently. They use statically pooled eGeMAPS features and also log to MLflow.

```bash
# Baseline 1: Linear Regression
python -m src.models.linear_baseline

# Baseline 2: Decision Tree / XGBoost
python -m src.models.tree_baseline
```

Hyperparameters (e.g., `max_depth`, `min_samples_leaf`) are set in `configs/model/tree.yaml`.

---

## 7. Visualizations

Publication-ready charts are generated automatically at the end of evaluation and logged to MLflow as high-DPI PNG artifacts. They can also be generated on demand:

```bash
# Predicted vs. human score scatter plots (phoneme, word, sentence)
python -m src.visualization.scatter --run-id <mlflow-run-id>

# Training / validation loss curves
python -m src.visualization.loss_curves --run-id <mlflow-run-id>

# Attention weight heatmap for a selected sentence
python -m src.visualization.attention --run-id <mlflow-run-id>

# Fairness bar charts (Children vs. Adults)
python -m src.visualization.fairness_charts --run-id <mlflow-run-id>
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
- **Git tagging**: Tag the commit corresponding to your thesis-final experiment for long-term traceability:
  ```bash
  git tag -a thesis-final -m "Final experiment run"
  git push origin thesis-final
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
│   │   ├── align_dataset.py    # Native dataset TextGrid parser
│   │   ├── extract.py          # openSMILE eGeMAPS LLD extraction
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
