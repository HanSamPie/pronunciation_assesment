# APA Project — Task List

## Phase 0 — Environment Setup & Project Scaffold

- `[x]` Create project directory structure
- `[x]` Create `requirements.txt`
- `[x]` Create `.gitignore`
- `[x]` Create `configs/base.yaml`, `configs/model/bigru.yaml`, `configs/model/linear.yaml`, `configs/model/tree.yaml`, `configs/experiment/default.yaml`

---

## Phase 1 — Data Preprocessing Pipeline

- `[x]` Implement `src/data/split.py` — speaker-independent 70/15/15 split
- `[x]` Implement `src/data/align.py` — MFA forced alignment wrapper (`english_mfa`)
- `[x]` Implement `src/data/extract.py` — openSMILE eGeMAPS extraction + boundary jitter
- `[x]` Implement `src/data/persist.py` — HDF5 serialization
- `[x]` Implement `src/data/normalize.py` — StandardScaler fit/transform/save

---

## Phase 2 — Model Implementation

- `[x]` Implement `src/models/bigru.py` — Hierarchical Multi-Task Bi-GRU
- `[ ]` Update Bi-GRU prediction heads for multi-metric support (Stress, Fluency, etc.)
- `[x]` Implement `src/training/loss.py` — weighted multi-task MSE loss
- `[x]` Implement `src/models/linear_baseline.py` — Linear Regression baseline
- `[x]` Implement `src/models/tree_baseline.py` — Decision Tree / XGBoost baseline

---

## Phase 3 — Scientific Logging & Configuration

- `[ ]` Implement `src/training/trainer.py` — MLflow training loop
- `[ ]` Create configuration for "all metrics" vs "major scores" support

---

## Phase 4 — Evaluation & Result Cache

- `[ ]` Implement `src/evaluation/evaluate.py` — PCC, RMSE, SRC metrics
- `[ ]` Implement `src/evaluation/cache.py` — SQLite MD5 result cache
- `[ ]` Implement `src/evaluation/fairness.py` — stratified fairness analysis

---

## Phase 5 — Visualizations

- `[ ]` Implement `src/visualization/scatter.py`
- `[ ]` Implement `src/visualization/loss_curves.py`
- `[ ]` Implement `src/visualization/attention.py`
- `[ ]` Implement `src/visualization/fairness_charts.py`
