# Automated Pronunciation Assessment (APA) — Implementation Plan

## Background

This project builds an **Automated Pronunciation Assessment (APA)** system trained on the **Speechocean762** dataset (5000 English sentences, Mandarin L1 speakers). The primary deliverable is a scientific artifact suitable for a thesis, requiring strict reproducibility, clean experiment tracking, and publication-quality visualizations.

The system predicts pronunciation quality at three granularities — **phoneme**, **word**, and **sentence** level — and compares a deep learning model against two classical baselines.

---

## User Review Required

> [!IMPORTANT]
> **Speaker-independent data splits** are the cornerstone of this project. All downstream work assumes splits are finalized before any feature extraction begins. If the split ratios (70/15/15) or the splitting strategy need adjustment, it must happen before Phase 1 begins.

> [!WARNING]
> **Boundary Jittering** is disabled by default (`use_boundary_jitter: false` in config). Enable explicitly for benchmarking runs only. This flag must be logged to MLflow so experiments with and without jittering remain distinguishable.

> [!NOTE]
> **Multi-task loss weights** are locked at `phoneme: 1.0 / word: 2.0 / sentence: 5.0`. These are defined as top-level fields in `configs/base.yaml` and logged to MLflow on every run, making them trivially editable and fully reproducible.

---

## Proposed Project Structure

```
pronunciation_assesment/
├── configs/                   # Hydra YAML configurations
│   ├── base.yaml
│   ├── model/
│   │   ├── bigru.yaml
│   │   ├── linear.yaml
│   │   └── tree.yaml
│   └── experiment/
│       └── default.yaml
├── data/
│   ├── raw/                   # Original Speechocean762 dataset
│   ├── splits/                # Speaker-independent split manifests (CSV/JSON)
│   ├── features/              # Serialized eGeMAPS features (HDF5/Parquet)
│   └── scalers/               # Saved StandardScaler objects
├── src/
│   ├── data/
│   │   ├── split.py           # Speaker-independent splitting logic
│   │   ├── align.py           # MFA forced alignment wrapper
│   │   ├── extract.py         # openSMILE eGeMAPS extraction
│   │   ├── persist.py         # HDF5/Parquet serialization
│   │   └── normalize.py       # StandardScaler fit/transform
│   ├── models/
│   │   ├── bigru.py           # Hierarchical Multi-Task Bi-GRU
│   │   ├── linear_baseline.py # Linear Regression baseline
│   │   └── tree_baseline.py   # Decision Tree / XGBoost baseline
│   ├── training/
│   │   ├── trainer.py         # Training loop with MLflow logging
│   │   └── loss.py            # Weighted multi-task loss function
│   ├── evaluation/
│   │   ├── evaluate.py        # Deterministic evaluation script
│   │   ├── cache.py           # SQLite-backed MD5 hash cache
│   │   └── fairness.py        # Stratified fairness analysis
│   └── visualization/
│       ├── scatter.py         # Predicted vs. human scatter plots
│       ├── loss_curves.py     # Train/validation loss curves
│       ├── attention.py       # Attention weight heatmaps
│       └── fairness_charts.py # Fairness bar charts
├── notebooks/                 # Exploratory / analysis notebooks
├── mlruns/                    # MLflow local tracking directory
├── results.db                 # SQLite evaluation cache
└── requirements.txt
```

---

## Proposed Changes

### Phase 1 — Data Preprocessing Pipeline

#### [NEW] `src/data/split.py`
Speaker-independent train/val/test split (70/15/15) by `speaker_id`. Outputs manifest files (CSV/JSON) keyed by speaker and sentence identifiers.

#### [NEW] `src/data/align.py`
Wrapper around the **Montreal Forced Aligner (MFA)** CLI using the **`english_mfa`** pre-trained acoustic model. Consumes dataset audio + transcripts, outputs TextGrid files with phoneme boundary timings ($t_{start}$, $t_{end}$). The model name is specified in `configs/base.yaml` as `mfa_model: english_mfa`.

#### [MODIFY] `src/data/extract.py`
Wrapper around **openSMILE** to extract continuous **eGeMAPS Low-Level Descriptors (LLDs, 23 features per frame)** from the full audio recording, and sequentially align these frames to the MFA-derived phoneme boundaries. Replaces the previous 88-functional per-phoneme extraction approach. Supports optional **boundary jittering** (±5 ms, controlled by `use_boundary_jitter` config flag).

#### [NEW] `src/data/persist.py`
Serializes the extracted feature tensors to **HDF5** (via `h5py`), keyed by `(speaker_id, sentence_id, phoneme_index)`. Ensures exact reproducibility by decoupling extraction from training.

#### [NEW] `src/data/normalize.py`
Fits a `sklearn.preprocessing.StandardScaler` on the training partition only. Applies the fitted scaler to val/test. Saves the scaler object via `joblib` for later use in evaluation. 

---

### Phase 2 — Model Specifications

#### [NEW] `src/models/bigru.py` — Primary: Hierarchical Multi-Task Bi-GRU

| Layer | Detail |
|---|---|
| Input | Sequence of aligned 23-dim eGeMAPS LLD frames per phoneme |
| Encoder | Bi-directional GRU |
| Regularization | Dropout + L2 weight decay |
| Pooling | Attention pooling (2 separate heads: word-level, sentence-level) |
| Phoneme head | MLP from raw Bi-GRU hidden states |
| Word head | MLP from word-mean aggregated states |
| Sentence head | MLP from sentence-mean aggregated states |
| Output activation | Sigmoid × max-score-for-metric |

#### [NEW] `src/training/loss.py` — Weighted Multi-Task Loss
Implements MSE-based multi-task loss with explicit gradient weighting across phoneme / word / sentence levels. Weights are read from config and default to:

```yaml
loss_weights:
  phoneme: 1.0
  word:    2.0
  sentence: 5.0
```

All three values are logged to MLflow at run initialization, making them trivially editable per experiment.

#### [NEW] `src/models/linear_baseline.py` — Baseline 1: Linear Regression
`sklearn.linear_model.LinearRegression` on statically pooled eGeMAPS features.

#### [NEW] `src/models/tree_baseline.py` — Baseline 2: Regression Tree / XGBoost
`sklearn.tree.DecisionTreeRegressor` or `xgboost.XGBRegressor` with tuned max-depth and min-samples-leaf to prevent overfitting on the 5000-sample dataset.

---

### Phase 3 — Scientific Logging & Configuration Management

#### [NEW] `configs/base.yaml` and `configs/experiment/default.yaml`
**Hydra**-managed YAML configuration files. Key locked-in defaults:

```yaml
# Reproducibility
seed: 42

# Data
mfa_model: english_mfa
feature_store: hdf5
use_boundary_jitter: false

# Loss weights (easily editable)
loss_weights:
  phoneme: 1.0
  word:    2.0
  sentence: 5.0

# Score Modeling Strategy
# Toggle between "major_scores" (Accuracy only) and "all_metrics" (all axes)
score_mode: major_scores  # default

# Definitions:
# - major_scores:
#     - Phoneme: Accuracy (0-2)
#     - Word:    Accuracy (0-10)
#     - Sentence: Accuracy (0-10)
# - all_metrics:
#     - Phoneme: Accuracy (0-2)
#     - Word:    Accuracy (0-10), Stress (5 or 10)
#     - Sentence: Accuracy (0-10), Completeness (0-1), Fluency (0-10), Prosodic (0-10)
```

#### [NEW] `src/training/trainer.py` — MLflow Training Loop
- Initializes an **MLflow** tracking run.
- Auto-logs: Git commit hash, parsed YAML config, per-epoch train/val loss, global random seeds.
- Registers artifacts: final model weights, fitted `StandardScaler`, evaluation reports.

---

### Phase 4 — Evaluation Script & Result Cache

#### [NEW] `src/evaluation/evaluate.py`
Deterministic evaluation computing **PCC (Pearson)**, **RMSE**, and **SRC (Spearman)** at phoneme, word, and sentence levels.

#### [NEW] `src/evaluation/cache.py`
MD5 hash = `hash(config) + hash(model weights) + hash(test dataset)`.  
Query a **SQLite** database (`results.db`) before running evaluation. Cache hit → load. Cache miss → evaluate and insert.

#### [NEW] `src/evaluation/fairness.py`
Stratifies evaluation metrics by demographic groups (child vs. adult) as annotated in Speechocean762. Reports RMSE and PCC per group.

---

### Phase 5 — Publication-Quality Visualizations

All charts are logged directly to **MLflow** as artifacts and saved as high-DPI PNGs.

#### [NEW] `src/visualization/scatter.py`
Predicted vs. human score scatter plots for phoneme, word, and sentence levels. Includes the identity line ($y = x$) and the PCC value in the legend.

#### [NEW] `src/visualization/loss_curves.py`
Training and validation loss curves per epoch to demonstrate convergence and confirm absence of overfitting.

#### [NEW] `src/visualization/attention.py`
Attention weight heatmaps: a specific sentence with aligned phonemes on the x-axis, attention head weights as the color intensity.

#### [NEW] `src/visualization/fairness_charts.py`
Grouped bar charts of RMSE and PCC for Children vs. Adults to reveal model bias.

#### [NEW] `src/visualization/mfa_alignment.py`
Compares the MFA forced alignment data against the dataset's ground truth. Evaluates:
- **Phoneme accuracy**: How accurately the phonemes were detected compared to the dataset.
- **Timestep accuracy**: The precision of the phoneme boundaries (start and end times).
Generates charts that visualize both phoneme and timestep accuracy relative to the accuracy score of the phrase.

---

## Technology Stack

| Purpose | Library / Tool |
|---|---|
| Forced Alignment | Montreal Forced Aligner (MFA) |
| Acoustic Features | openSMILE + eGeMAPS config |
| Feature Storage | `h5py` (HDF5) |
| Normalization | `scikit-learn` StandardScaler |
| Primary Model | `PyTorch` |
| Baselines | `scikit-learn`, `xgboost` |
| Config Management | `Hydra` + `omegaconf` |
| Experiment Tracking | `MLflow` |
| Evaluation Cache | `sqlite3` (stdlib) |
| Visualizations | `matplotlib`, `seaborn` |
| Reproducibility | Seeds, Git commit logging, scaler/model artifact registration |

---

## Locked Design Decisions

| Decision | Value |
|---|---|
| Multi-task loss weights | `phoneme: 1.0 / word: 2.0 / sentence: 5.0` (in `configs/base.yaml`) |
| Feature storage format | **HDF5** (via `h5py`) |
| Boundary jittering default | **`false`** (opt-in, not opt-out) |
| MFA acoustic model | **`english_mfa`** |

---

## Verification Plan

### Automated Tests
- Unit test `split.py`: verify no `speaker_id` appears in more than one partition.
- Unit test `normalize.py`: verify scaler is fit on train only (test/val means are non-zero after transform).
- Unit test `cache.py`: verify cache hit/miss logic with mock hashes.
- Integration test: run full pipeline on a small synthetic subset (10 speakers, 20 sentences).

### MLflow Verification
- After a training run, confirm that MLflow UI shows: config YAML, git hash, per-epoch loss curves, and all registered artifacts.

### Evaluation Verification
- Run evaluation twice on the same model; confirm SQLite cache is hit on the second run and results are byte-identical.

### Visual Inspection
- Manual review of all 4 chart types for a reference run before thesis submission.
