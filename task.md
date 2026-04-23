# APA Project — Task List

## Phase 0 — Environment Setup & Project Scaffold

- `[x]` Create project directory structure
- `[x]` Create `requirements.txt`
- `[x]` Create `.gitignore`
- `[x]` Create `configs/base.yaml`, `configs/model/bigru.yaml`, `configs/model/linear.yaml`, `configs/model/tree.yaml`, `configs/experiment/default.yaml`
- `[x]` Create `Makefile` for easily manageable project workflows (test, train, eval)
- `[x]` Create `pyproject.toml` / `setup.py` for easily manageable `src` imports and testing

---

## Phase 1 — Data Preprocessing Pipeline

- `[x]` Implement `src/data/split.py` — speaker-independent 70/15/15 split
- `[x]` Implement `src/data/align.py` — MFA forced alignment wrapper (`english_mfa`)
- `[ ]` Refactor `src/data/extract.py` — continuous openSMILE eGeMAPS LLD extraction (23 features/frame) + phoneme boundary alignment
- `[ ]` Update `src/data/persist.py` — handle variable-length LLD frame sequences in HDF5 serialization
- `[ ]` Update `src/data/normalize.py` — StandardScaler fit/transform for 23-dim LLD frames

---

## Phase 2 — Model Implementation

- `[x]` Implement `src/models/bigru.py` — Hierarchical Multi-Task Bi-GRU
- `[x]` Update Bi-GRU prediction heads for multi-metric support (Stress, Fluency, etc.)
- `[x]` Implement `src/training/loss.py` — weighted multi-task MSE loss
- `[x]` Implement `src/models/linear_baseline.py` — Linear Regression baseline
- `[x]` Implement `src/models/tree_baseline.py` — Decision Tree / XGBoost baseline

---

## Phase 3 — Scientific Logging & Configuration

- `[x]` Implement `src/training/trainer.py` — MLflow training loop
- `[x]` Create configuration for "all metrics" vs "major scores" support

---

## Phase 4 — Evaluation & Result Cache

- `[x]` Implement `src/evaluation/evaluate.py` — PCC, RMSE, SRC metrics
- `[x]` Implement `src/evaluation/cache.py` — SQLite MD5 result cache
- `[x]` Implement `src/evaluation/fairness.py` — stratified fairness analysis

---

## Phase 5 — Visualizations

- `[ ]` Implement `src/visualization/scatter.py`
- `[ ]` Implement `src/visualization/loss_curves.py`
- `[ ]` Implement `src/visualization/attention.py`
- `[ ]` Implement `src/visualization/fairness_charts.py`
- `[ ]` Implement `src/visualization/mfa_alignment.py` — Compare MFA phoneme/timestep accuracy against dataset ground truth and plot against phrase accuracy scores

---

## Phase 6 — Verification & Testing

- `[ ]` Implement unit tests for `src/data/split.py` (verify no `speaker_id` leakage)
- `[ ]` Implement unit tests for `src/data/normalize.py` (verify scaler fit only on train)
- `[ ]` Implement unit tests for `src/evaluation/cache.py` (verify cache hit/miss logic)
- `[ ]` Implement integration tests (run full pipeline on small synthetic subset)
- `[ ]` Verify MLflow logging (config, git hash, loss curves, artifacts)
- `[ ]` Verify Evaluation caching logic
- `[ ]` Visual inspection of generated charts

---

## Phase 7 — Finalization

- `[ ]` Integrate baseline models with MLflow tracking
- `[ ]` Create exploratory data analysis notebooks (`notebooks/`)
- `[ ]` Tag the `thesis-final` commit once all experiments are complete
