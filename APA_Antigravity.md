Project Specification: Automated Pronunciation Assessment (Antigravity)

1. Data Preprocessing Pipeline

Strict separation of Train, Validation, and Test sets must be enforced prior to any feature extraction to prevent data leakage.

Dataset: Speechocean762 (5000 English sentences, Mandarin L1 speakers).

Data Splitting Strategy (Speaker-Independent):

Split the dataset into Train (e.g., 70%), Validation (15%), and Test (15%) partitions strictly by speaker_id. A speaker present in the training set must not appear in the validation or test sets.

Step 1: Forced Alignment:

Tool: Montreal Forced Aligner (MFA).

Output: Phonetic boundaries ($t_{start}$, $t_{end}$) for each phoneme based on the provided transcripts.

Step 2: Acoustic Feature Extraction:

Tool: openSMILE.

Feature Set: eGeMAPS (88 features).

Process: Extract features strictly within the bounds of ($t_{start}$, $t_{end}$) for each phoneme.

Step 3: Feature Persistence:

Serialize and store the extracted raw eGeMAPS features locally (e.g., using HDF5 or Parquet) keyed by speaker_id and sentence_id to avoid recomputation and ensure exact reproducibility.

Step 4: Normalization:

Apply standard scaling (zero mean, unit variance) to the eGeMAPS features.

Fit the scaler only on the training set partition. Apply the fitted scaler to the validation and test sets. Store the scaler object.

2. Model Specifications

2.1 Primary Model: Hierarchical Multi-Task Bi-GRU

Input: Sequence of 88-dimensional eGeMAPS feature vectors per phoneme.

Data Augmentation (Optional/Configurable):

Boundary Jittering: During training, randomly shift ($t_{start}$, $t_{end}$) by $x \in [-5\text{ms}, +5\text{ms}]$ and re-pool/re-extract features to build robustness against MFA inaccuracies. Controlled via configuration flag use_boundary_jitter.

Encoder: Bi-directional Gated Recurrent Unit (Bi-GRU).

Regularization: Dropout, L2 Weight Decay to mitigate memorization of specific speakers.

Pooling Layer: Attention Pooling.

Two separate attention heads: One for word-level aggregation, one for sentence-level aggregation.

Prediction Heads (MLPs):

Phoneme-level: Takes raw Bi-GRU hidden states.

Word-level: Takes word-mean aggregated states.

Sentence-level: Takes sentence-mean aggregated states.

Activation: Sigmoid, scaled by the maximum possible score for the respective metric.

Loss Function: Weighted multi-task loss (e.g., Mean Squared Error) to balance gradient updates. The weighting factors must explicitly account for both the differing target metric scales and the severe imbalance in sample frequencies (i.e., gradients from the abundant phoneme samples must not overpower the sparse sentence-level samples).

2.2 Baseline 1: Linear Regression

Input: Flattened or statically pooled eGeMAPS features.

Implementation: scikit-learn LinearRegression.

2.3 Baseline 2: Regression Tree

Input: Flattened or statically pooled eGeMAPS features.

Implementation: scikit-learn DecisionTreeRegressor or xgboost.XGBRegressor.

Hyperparameters: Max depth, minimum samples per leaf (to prevent overfitting on the 5000-sample dataset).

3. Scientific Logging and Configuration Management

To guarantee reproducibility, all experiments must be strictly version-controlled using MLflow.

Configuration: Use YAML files (e.g., via Hydra) to define all hyperparameters, seed values, data paths, and flags (like use_boundary_jitter).

Experiment Tracking (MLflow):

Initialize an MLflow tracking server (local or remote).

Automatically log: Git commit hash, parsed YAML configuration, training/validation loss per epoch, and global random seeds.

Artifacts: Register the final model weights, the fitted StandardScaler object, and evaluation reports.

4. Evaluation Script and Result Cache

Evaluation must be deterministic and efficient.

Metrics: Pearson Correlation Coefficient (PCC), Root Mean Squared Error (RMSE), Spearman’s Rank Correlation (SRC).

Caching Mechanism:

Generate an MD5 hash representing the combination of: Config Hash + Model Weights Hash + Test Dataset Hash.

Store results in a local SQLite database mapped to this hash.

Before running evaluation, query the SQLite database. If the hash exists, load results; if not, run evaluation and insert into the database.

Fairness Analysis: Stratify evaluation metrics by demographic groups (e.g., child vs. adult speakers) provided by Speechocean762.

5. Required Visualizations for Thesis

The evaluation script should automatically generate the following publication-ready charts (e.g., using matplotlib/seaborn), logged directly into MLflow:

Correlation Scatter Plots: Predicted vs. Human scores for Phoneme, Word, and Sentence levels. Include the identity line ($y=x$) and the PCC value in the legend.

Training/Validation Loss Curves: Demonstrating convergence and confirming the absence of overfitting.

Attention Weight Heatmaps: Qualitative visualization showing a specific sentence, aligned phonemes, and attention weights assigned by the model.

Fairness Bar Charts: RMSE and PCC grouped by demographic (Children vs. Adults) to show model bias.
