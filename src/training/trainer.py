"""
MLflow training loop for hierarchical pronunciation assessment models.

This module serves as both a library (``run_training``) and a CLI entrypoint
(``python -m src.training.trainer`` / ``make train``).

CLI usage
---------
Train a single model:
    make train MODEL=bigru
    make train MODEL=linear
    make train MODEL=tree

Train all models sequentially:
    make train              # MODEL defaults to "all"

The ``model`` config group is resolved via Hydra config-group override.  The
model name must match one of the YAML files in ``configs/model/``.

MLflow server
-------------
The entrypoint auto-detects whether a local MLflow tracking server is already
listening on ``cfg.mlflow_tracking_uri``.  If not, it spawns one as a
background subprocess and shuts it down after training completes.
"""

import logging
import socket
import subprocess
import sys
import tempfile
import time
from pathlib import Path
from typing import Optional

import mlflow
import numpy as np
import torch
import torch.nn as nn
from omegaconf import DictConfig, OmegaConf
from torch.utils.data import DataLoader, Dataset

log = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Git helpers
# ---------------------------------------------------------------------------


def _get_git_hash() -> str:
    """Retrieve the current Git commit hash."""
    try:
        return subprocess.check_output(
            ["git", "rev-parse", "--short", "HEAD"],
            stderr=subprocess.STDOUT,
            text=True,
        ).strip()
    except Exception:
        return "unknown"


# ---------------------------------------------------------------------------
# MLflow server management
# ---------------------------------------------------------------------------


def _uri_is_local_http(uri: str) -> tuple[bool, str, int]:
    """Return (is_local_http, host, port) for a tracking URI like http://host:port."""
    if not uri.startswith("http://") and not uri.startswith("https://"):
        return False, "", 0
    try:
        from urllib.parse import urlparse
        parsed = urlparse(uri)
        host = parsed.hostname or "127.0.0.1"
        port = parsed.port or 5000
        return True, host, port
    except Exception:
        return False, "", 0


def _port_open(host: str, port: int, timeout: float = 1.0) -> bool:
    """Return True if a TCP connection to host:port succeeds."""
    try:
        with socket.create_connection((host, port), timeout=timeout):
            return True
    except OSError:
        return False


def _ensure_mlflow_server(tracking_uri: str) -> Optional[subprocess.Popen]:
    """
    Ensure an MLflow tracking server is reachable.

    If the tracking URI points to a local HTTP server that is not yet
    running, start one and wait up to 15 s for it to become available.

    Parameters
    ----------
    tracking_uri : str
        MLflow tracking URI from config (e.g. ``http://127.0.0.1:5000``).

    Returns
    -------
    subprocess.Popen or None
        The process handle if a new server was started, else None.
        The caller is responsible for terminating the process.
    """
    is_local, host, port = _uri_is_local_http(tracking_uri)
    if not is_local:
        # Remote or file-based URI — nothing to manage
        mlflow.set_tracking_uri(tracking_uri)
        return None

    mlflow.set_tracking_uri(tracking_uri)

    if _port_open(host, port):
        log.info("MLflow server already running at %s", tracking_uri)
        return None

    log.info("Starting MLflow tracking server at %s …", tracking_uri)
    proc = subprocess.Popen(
        [
            sys.executable, "-m", "mlflow", "server",
            "--host", host,
            "--port", str(port),
        ],
        stdout=subprocess.DEVNULL,
        stderr=subprocess.DEVNULL,
    )

    # Wait for the server to become ready (up to 15 s)
    deadline = time.time() + 15.0
    while time.time() < deadline:
        if _port_open(host, port):
            log.info("MLflow server ready at %s", tracking_uri)
            return proc
        time.sleep(0.5)

    # Server didn't start in time — kill it and fall back to file-based tracking
    proc.terminate()
    log.warning(
        "MLflow server failed to start within 15 s. "
        "Falling back to file-based tracking (mlruns/)."
    )
    mlflow.set_tracking_uri("mlruns")
    return None


# ---------------------------------------------------------------------------
# Dataset — wraps the pre-computed HDF5 feature + scaler pipeline
# ---------------------------------------------------------------------------


class PronunciationDataset(Dataset):
    """
    PyTorch Dataset backed by pre-computed, normalized eGeMAPS features
    with hierarchical ground-truth scores from Speechocean762.

    Each item represents one *sentence* and contains:

    * ``phoneme_features`` — tensor ``(T, n_lld)``
    * ``phoneme_mask``     — boolean tensor ``(T,)``
    * ``word_boundaries``  — list of ``(start, end)`` tuples derived from
      the word→phoneme grouping in ``scores.json``
    * ``targets``          — dict of score tensors at the correct granularity:

      - ``phoneme_*`` metrics → tensor ``(T,)``   one per phoneme
      - ``word_*``    metrics → tensor ``(W,)``   one per word
      - ``sentence_*``metrics → scalar tensor

    Parameters
    ----------
    feature_arrays : dict[(spk, sent, ph_idx) -> np.ndarray]
        Scaled LLD frame arrays from ``normalize.transform_arrays``.
    meta_df : pd.DataFrame
        Companion metadata (output of ``persist.load_meta``).
    cfg : DictConfig
        Full config (used to resolve active metrics and score ranges).
    hierarchical_scores : dict[str, dict]
        Loaded ``scores.json`` keyed by 9-digit sentence ID.
    score_df : pd.DataFrame, optional
        Manifest DataFrame with ``speaker_id``, ``sentence_id``, ``_split``.
        Used only to gate which sentences have labels.
    """

    def __init__(
        self,
        feature_arrays,
        meta_df,
        cfg,
        hierarchical_scores: dict[str, dict],
        score_df=None,
    ):
        from omegaconf import OmegaConf
        from src.data.scores import (
            flat_phoneme_scores,
            word_boundaries_from_entry,
            word_scores,
        )

        self.cfg = cfg
        self.score_mode: str = str(cfg.score_mode)
        self.metric_max_scores: dict[str, float] = {
            k: float(v)
            for k, v in OmegaConf.to_container(
                cfg.metrics[self.score_mode], resolve=True
            ).items()
        }
        self.active_metrics = list(self.metric_max_scores.keys())

        # Classify active metrics by level for target construction
        self._phoneme_metrics = [m for m in self.active_metrics if m.startswith("phoneme_")]
        self._word_metrics    = [m for m in self.active_metrics if m.startswith("word_")]
        self._sentence_metrics = [m for m in self.active_metrics if m.startswith("sentence_")]

        # Mapping from canonical metric name → scores.json key
        # (strip the level prefix to get the raw key in scores.json)
        _SENTENCE_KEY_MAP: dict[str, str] = {
            "sentence_accuracy":     "accuracy",
            "sentence_fluency":      "fluency",
            "sentence_prosodic":     "prosodic",
        }
        _WORD_KEY_MAP: dict[str, str] = {
            "word_accuracy": "accuracy",
            "word_stress":   "stress",
        }

        # Group phoneme arrays by (speaker_id, sentence_id)
        sentences: dict[tuple, list] = {}
        for (spk, sent, ph_idx), arr in sorted(
            feature_arrays.items(), key=lambda x: (x[0][0], x[0][1], x[0][2])
        ):
            key = (spk, sent)
            if key not in sentences:
                sentences[key] = []
            sentences[key].append((ph_idx, arr))

        # Build valid sentence ID set from score_df (if provided)
        valid_sents: set[tuple[str, str]] | None = None
        if score_df is not None:
            valid_sents = set(
                zip(
                    score_df["speaker_id"].astype(str),
                    score_df["sentence_id"].astype(str),
                )
            )

        skipped = 0
        self.samples = []
        for (spk, sent), ph_list in sentences.items():
            # Gate: only include sentences present in the score manifest
            if valid_sents is not None and (str(spk), str(sent)) not in valid_sents:
                skipped += 1
                continue

            # Look up hierarchical scores for this sentence
            entry = hierarchical_scores.get(str(sent).zfill(9))
            if entry is None:
                skipped += 1
                continue

            ph_list.sort(key=lambda x: x[0])  # sort by phoneme index
            arrays = [arr for _, arr in ph_list]

            # Mean-pool each phoneme's frames → one vector per phoneme
            pooled = [arr.mean(axis=0) for arr in arrays]  # list of (n_lld,)
            T = len(pooled)

            # --- Word boundaries & phoneme-level targets ---
            # The TextGrid-based extraction may produce a different phoneme
            # count than scores.json (different inventories, schwa splits,
            # silence handling).  We handle both cases:
            json_word_bounds = word_boundaries_from_entry(entry)
            n_phones_in_scores = sum(e - s for s, e in json_word_bounds)
            n_words = len(json_word_bounds)

            if n_phones_in_scores == T:
                # --- Exact match: use granular per-phoneme scores ---
                word_boundaries = json_word_bounds
                phone_accs = flat_phoneme_scores(entry)
            else:
                # --- Mismatch: approximate word boundaries & broadcast ---
                # Distribute HDF5 phonemes across words as evenly as
                # possible, then broadcast each word's accuracy to its
                # phoneme slots.
                if n_words > 0 and T > 0:
                    base_per_word = T // n_words
                    remainder = T % n_words
                    word_boundaries = []
                    offset = 0
                    for w_idx in range(n_words):
                        w_len = base_per_word + (1 if w_idx < remainder else 0)
                        word_boundaries.append((offset, offset + w_len))
                        offset += w_len
                else:
                    word_boundaries = [(0, T)]

                # Broadcast word accuracy to each phoneme
                w_accs = word_scores(entry, "accuracy")
                phone_accs = []
                for w_idx, (ws, we) in enumerate(word_boundaries):
                    w_acc = w_accs[w_idx] if w_idx < len(w_accs) else 0.0
                    # Scale word accuracy (0-10) → phoneme range (0-2)
                    phone_accs.extend([w_acc / 5.0] * (we - ws))

            # --- Targets at each level ---
            targets: dict[str, list[float] | float] = {}

            # Phoneme-level targets
            if self._phoneme_metrics:
                for metric in self._phoneme_metrics:
                    if metric == "phoneme_accuracy":
                        targets[metric] = phone_accs
                    else:
                        targets[metric] = [0.0] * T

            # Word-level targets
            if self._word_metrics:
                for metric in self._word_metrics:
                    raw_key = _WORD_KEY_MAP.get(metric)
                    if raw_key:
                        w_scores = word_scores(entry, raw_key)
                        # Pad or truncate to match actual word boundary count
                        n_wb = len(word_boundaries)
                        if len(w_scores) >= n_wb:
                            targets[metric] = w_scores[:n_wb]
                        else:
                            targets[metric] = w_scores + [0.0] * (n_wb - len(w_scores))
                    else:
                        targets[metric] = [0.0] * len(word_boundaries)

            # Sentence-level targets
            for metric in self._sentence_metrics:
                raw_key = _SENTENCE_KEY_MAP.get(metric)
                if raw_key and raw_key in entry:
                    targets[metric] = float(entry[raw_key])
                else:
                    targets[metric] = 0.0

            self.samples.append(
                {
                    "spk": spk,
                    "sent": sent,
                    "pooled": pooled,
                    "word_boundaries": word_boundaries,
                    "targets": targets,
                }
            )

        if skipped:
            log.info(
                "Skipped %d sentences (missing scores or phoneme count mismatch).",
                skipped,
            )

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, idx: int) -> dict:
        item = self.samples[idx]
        pooled = item["pooled"]  # list of (n_lld,) arrays
        T = len(pooled)
        n_lld = pooled[0].shape[0]

        phoneme_features = torch.from_numpy(
            np.stack(pooled, axis=0).astype(np.float32)
        )  # (T, n_lld)
        phoneme_mask = torch.ones(T, dtype=torch.bool)

        targets: dict[str, torch.Tensor] = {}
        for k, v in item["targets"].items():
            if isinstance(v, list):
                targets[k] = torch.tensor(v, dtype=torch.float32)
            else:
                targets[k] = torch.tensor(v, dtype=torch.float32)

        return {
            "phoneme_features": phoneme_features,
            "phoneme_mask": phoneme_mask,
            "word_boundaries": item["word_boundaries"],
            "targets": targets,
        }


def _collate_fn(batch: list[dict]) -> dict:
    """Pad variable-length phoneme sequences and collate hierarchical targets.

    Target collation strategy mirrors the model output shapes:

    * **phoneme-level** (``phoneme_*``) — concatenate valid phonemes across
      the batch → ``(N_valid_phonemes,)``
    * **word-level** (``word_*``) — concatenate words across the batch →
      ``(N_words_in_batch,)``
    * **sentence-level** (``sentence_*``) — stack scalars → ``(B,)``
    """
    max_t = max(item["phoneme_features"].shape[0] for item in batch)
    n_lld = batch[0]["phoneme_features"].shape[1]

    padded_features = torch.zeros(len(batch), max_t, n_lld)
    padded_mask = torch.zeros(len(batch), max_t, dtype=torch.bool)

    for i, item in enumerate(batch):
        t = item["phoneme_features"].shape[0]
        padded_features[i, :t, :] = item["phoneme_features"]
        padded_mask[i, :t] = item["phoneme_mask"]

    word_boundaries = [item["word_boundaries"] for item in batch]

    # Collate targets by hierarchy level
    target_keys = list(batch[0]["targets"].keys())
    targets: dict[str, torch.Tensor] = {}
    for k in target_keys:
        if k.startswith("phoneme_"):
            # Concatenate per-phoneme targets → (N_valid_phonemes,)
            targets[k] = torch.cat([item["targets"][k] for item in batch])
        elif k.startswith("word_"):
            # Concatenate per-word targets → (N_words_in_batch,)
            targets[k] = torch.cat([item["targets"][k] for item in batch])
        else:
            # sentence-level: stack scalars → (B,)
            targets[k] = torch.stack([item["targets"][k] for item in batch])

    return {
        "phoneme_features": padded_features,
        "phoneme_mask": padded_mask,
        "word_boundaries": word_boundaries,
        "targets": targets,
    }


# ---------------------------------------------------------------------------
# Score loading helpers
# ---------------------------------------------------------------------------


def _load_scores(cfg: DictConfig):
    """
    Load split manifests and hierarchical scores for training.

    Returns
    -------
    score_df : pd.DataFrame
        Merged manifest with ``_split`` column and zero-padded IDs.
    hierarchical_scores : dict[str, dict]
        ``scores.json`` keyed by 9-digit sentence ID.
    """
    import pandas as pd
    from src.data.scores import load_hierarchical_scores

    splits_dir = Path(cfg.splits_dir)
    dfs = []
    for partition in ("train", "val", "test"):
        csv_path = splits_dir / f"{partition}_manifest.csv"
        if csv_path.exists():
            df = pd.read_csv(csv_path)
            df["_split"] = partition
            dfs.append(df)
    if not dfs:
        raise FileNotFoundError(
            f"No split manifests found in {splits_dir}. "
            "Run `make split` first."
        )
    merged = pd.concat(dfs, ignore_index=True)

    # Normalize IDs to zero-padded strings so they match HDF5 group keys
    # (speaker: 4-digit, sentence: 9-digit).
    merged["speaker_id"] = merged["speaker_id"].astype(str).str.zfill(4)
    merged["sentence_id"] = merged["sentence_id"].astype(str).str.zfill(9)

    # Load hierarchical scores from scores.json
    scores_path = Path(cfg.scores_file)
    hierarchical_scores = load_hierarchical_scores(scores_path)

    return merged, hierarchical_scores


def _build_dataloaders(
    cfg: DictConfig, batch_size: int
) -> tuple[DataLoader, DataLoader]:
    """
    Load scaled HDF5 features and ground-truth scores, return DataLoaders.

    Expects ``data/features/{train_scaled,val_scaled}_features.h5`` produced
    by ``make normalize``.

    Returns
    -------
    (train_loader, val_loader)
    """
    from src.data.persist import load_features

    score_df, hierarchical_scores = _load_scores(cfg)

    loaders = {}
    for partition in ("train", "val"):
        try:
            arrays, meta_df = load_features(cfg, split=f"{partition}_scaled")
        except FileNotFoundError:
            log.warning(
                "Scaled features not found for '%s'. "
                "Trying un-scaled features …", partition
            )
            arrays, meta_df = load_features(cfg, split=partition)

        split_scores = score_df[score_df["_split"] == partition].copy()

        ds = PronunciationDataset(
            feature_arrays=arrays,
            meta_df=meta_df,
            cfg=cfg,
            hierarchical_scores=hierarchical_scores,
            score_df=split_scores,
        )
        shuffle = partition == "train"
        loaders[partition] = DataLoader(
            ds,
            batch_size=batch_size,
            shuffle=shuffle,
            collate_fn=_collate_fn,
            num_workers=0,
        )
        log.info(
            "Loaded %d sentences for '%s' partition.",
            len(ds),
            partition,
        )

    return loaders["train"], loaders["val"]


# ---------------------------------------------------------------------------
# Core training loop (BiGRU / deep model)
# ---------------------------------------------------------------------------


def run_training(
    cfg: DictConfig,
    model: nn.Module,
    loss_fn: nn.Module,
    optimizer: torch.optim.Optimizer,
    train_loader: DataLoader,
    val_loader: DataLoader,
    device: torch.device,
    scaler_path: Optional[Path] = None,
    scheduler: Optional[torch.optim.lr_scheduler._LRScheduler] = None,
) -> Path:
    """
    Executes the training loop with MLflow tracking.

    Parameters
    ----------
    cfg : DictConfig
        Hydra configuration containing 'model.epochs', 'seed', etc.
    model : nn.Module
        The hierarchical model (e.g., HierarchicalBiGRU).
    loss_fn : nn.Module
        MultiTaskLoss instance.
    optimizer : torch.optim.Optimizer
        PyTorch optimizer.
    train_loader : DataLoader
        DataLoader yielding batches. Expected format is either a dictionary
        with keys: 'phoneme_features', 'phoneme_mask', 'word_boundaries',
        'targets', or a tuple matching that order.
    val_loader : DataLoader
        DataLoader for validation.
    device : torch.device
        Device to train on.
    scaler_path : Path, optional
        Path to the fitted StandardScaler artifact to register.
    scheduler : torch.optim.lr_scheduler._LRScheduler, optional
        Learning rate scheduler.  If provided, ``scheduler.step(val_loss)``
        is called at the end of each epoch (designed for ReduceLROnPlateau).

    Returns
    -------
    Path
        Path to the saved final model weights.
    """
    epochs = int(cfg.model.epochs)
    seed = int(cfg.seed)

    # --- Early stopping configuration ---
    es_cfg = cfg.model.get("early_stopping", None)
    es_patience: int = int(es_cfg.patience) if es_cfg else 0
    es_min_delta: float = float(es_cfg.min_delta) if es_cfg else 0.0
    best_val_loss: float = float("inf")
    epochs_without_improvement: int = 0
    best_state_dict: Optional[dict] = None

    model.to(device)
    loss_fn.to(device)

    # Collect parameters to log
    params: dict = {
        "seed": seed,
        "git_hash": _get_git_hash(),
    }

    if hasattr(model, "params_dict"):
        params.update(model.params_dict())
    if hasattr(loss_fn, "weights_dict"):
        params.update(loss_fn.weights_dict())

    experiment_name = cfg.get("experiment_name", "apa_default")
    mlflow.set_experiment(experiment_name)

    with mlflow.start_run() as run:
        log.info("Started MLflow run %s", run.info.run_id)

        # Log parameters
        mlflow.log_params(params)

        # Save and log the parsed YAML configuration
        with tempfile.NamedTemporaryFile("w", delete=False, suffix=".yaml") as f:
            f.write(OmegaConf.to_yaml(cfg))
            tmp_cfg_path = f.name
        mlflow.log_artifact(tmp_cfg_path, "config")
        Path(tmp_cfg_path).unlink()

        # Log the scaler artifact if provided
        if scaler_path and scaler_path.exists():
            mlflow.log_artifact(str(scaler_path), "scalers")

        for epoch in range(1, epochs + 1):
            # --- Training Phase ---
            model.train()
            train_loss_total = 0.0
            train_steps = 0

            for batch in train_loader:
                if isinstance(batch, dict):
                    features = batch["phoneme_features"].to(device)
                    mask = batch["phoneme_mask"].to(device)
                    word_bounds = batch["word_boundaries"]
                    targets = {k: v.to(device) for k, v in batch["targets"].items()}
                else:
                    features, mask, word_bounds, targets = batch
                    features = features.to(device)
                    mask = mask.to(device)
                    targets = {k: v.to(device) for k, v in targets.items()}

                optimizer.zero_grad()
                preds = model(features, mask, word_bounds)
                loss, component_losses = loss_fn(preds, targets)
                loss.backward()
                optimizer.step()

                train_loss_total += loss.item()
                train_steps += 1

            avg_train_loss = train_loss_total / max(1, train_steps)

            # --- Validation Phase ---
            model.eval()
            val_loss_total = 0.0
            val_steps = 0

            with torch.no_grad():
                for batch in val_loader:
                    if isinstance(batch, dict):
                        features = batch["phoneme_features"].to(device)
                        mask = batch["phoneme_mask"].to(device)
                        word_bounds = batch["word_boundaries"]
                        targets = {k: v.to(device) for k, v in batch["targets"].items()}
                    else:
                        features, mask, word_bounds, targets = batch
                        features = features.to(device)
                        mask = mask.to(device)
                        targets = {k: v.to(device) for k, v in targets.items()}

                    preds = model(features, mask, word_bounds)
                    loss, _ = loss_fn(preds, targets)

                    val_loss_total += loss.item()
                    val_steps += 1

            avg_val_loss = val_loss_total / max(1, val_steps)

            # Current learning rate (for logging)
            current_lr = optimizer.param_groups[0]["lr"]

            log.info(
                "Epoch %3d/%3d | Train Loss: %.4f | Val Loss: %.4f | LR: %.2e",
                epoch, epochs, avg_train_loss, avg_val_loss, current_lr,
            )

            mlflow.log_metrics(
                {
                    "train_loss": avg_train_loss,
                    "val_loss": avg_val_loss,
                    "learning_rate": current_lr,
                },
                step=epoch,
            )

            # --- Learning rate scheduler step ---
            if scheduler is not None:
                scheduler.step(avg_val_loss)

            # --- Early stopping check ---
            if es_patience > 0:
                if avg_val_loss < best_val_loss - es_min_delta:
                    best_val_loss = avg_val_loss
                    epochs_without_improvement = 0
                    best_state_dict = {
                        k: v.clone() for k, v in model.state_dict().items()
                    }
                else:
                    epochs_without_improvement += 1
                    log.info(
                        "  Early stopping: %d/%d epochs without improvement.",
                        epochs_without_improvement, es_patience,
                    )
                    if epochs_without_improvement >= es_patience:
                        log.info(
                            "Early stopping triggered at epoch %d. "
                            "Restoring best weights (val_loss=%.4f).",
                            epoch, best_val_loss,
                        )
                        if best_state_dict is not None:
                            model.load_state_dict(best_state_dict)
                        mlflow.log_metric("early_stop_epoch", epoch)
                        break

        # --- Finalization ---
        models_dir = Path("models")
        models_dir.mkdir(exist_ok=True)
        model_name = cfg.get("model", {}).get("name", "model") if hasattr(cfg, "model") else "model"
        model_save_path = models_dir / f"{model_name}_{run.info.run_id}.pth"
        torch.save(model.state_dict(), model_save_path)
        mlflow.log_artifact(str(model_save_path), "models")

        log.info("Training complete. Weights saved to %s", model_save_path)
        return model_save_path


# ---------------------------------------------------------------------------
# Per-model training dispatchers
# ---------------------------------------------------------------------------


def _train_bigru(cfg: DictConfig) -> None:
    """Build and train the HierarchicalBiGRU model."""
    from src.models.bigru import HierarchicalBiGRU
    from src.training.loss import MultiTaskLoss

    torch.manual_seed(int(cfg.seed))
    np.random.seed(int(cfg.seed))

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    log.info("Training BiGRU on device: %s", device)

    batch_size = int(cfg.model.batch_size)
    train_loader, val_loader = _build_dataloaders(cfg, batch_size)

    # Infer input feature dimension from the first batch
    sample_batch = next(iter(train_loader))
    n_lld = sample_batch["phoneme_features"].shape[-1]
    log.info("Feature dimensionality: %d", n_lld)

    model = HierarchicalBiGRU(cfg, input_size=n_lld)
    loss_fn = MultiTaskLoss(cfg)
    optimizer = torch.optim.Adam(
        model.parameters(),
        lr=float(cfg.model.learning_rate),
        weight_decay=float(cfg.model.l2_weight_decay),
    )

    # Learning rate scheduler (ReduceLROnPlateau on validation loss)
    scheduler = None
    sched_cfg = cfg.model.get("scheduler", None)
    if sched_cfg is not None:
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
            optimizer,
            mode="min",
            factor=float(sched_cfg.factor),
            patience=int(sched_cfg.patience),
            min_lr=float(sched_cfg.min_lr),
            verbose=True,
        )
        log.info(
            "LR scheduler: ReduceLROnPlateau(factor=%.2f, patience=%d, min_lr=%.1e)",
            float(sched_cfg.factor), int(sched_cfg.patience), float(sched_cfg.min_lr),
        )

    scaler_path = Path(cfg.scalers_dir) / "standard_scaler.joblib"
    run_training(
        cfg=cfg,
        model=model,
        loss_fn=loss_fn,
        optimizer=optimizer,
        train_loader=train_loader,
        val_loader=val_loader,
        device=device,
        scaler_path=scaler_path if scaler_path.exists() else None,
        scheduler=scheduler,
    )


def _train_sklearn_baseline(cfg: DictConfig, model_class) -> None:
    """
    Train a sklearn/XGBoost baseline model (LinearBaseline or TreeBaseline).

    Features are mean-pooled per sentence before fitting.  Baselines are
    single-output regressors, so they only use sentence-level targets.
    For granular metrics (phoneme/word), the baseline targets the mean
    of the granular ground-truth scores within the sentence.
    """
    import numpy as np
    from src.data.persist import load_features
    from omegaconf import OmegaConf

    np.random.seed(int(cfg.seed))

    active_metrics: list[str] = list(
        OmegaConf.to_container(cfg.metrics[cfg.score_mode], resolve=True).keys()
    )

    score_df, hierarchical_scores = _load_scores(cfg)

    def _pool_split(partition: str):
        """Load features and mean-pool per sentence → (X, y_dict)."""
        try:
            arrays, meta_df = load_features(cfg, split=f"{partition}_scaled")
        except FileNotFoundError:
            arrays, meta_df = load_features(cfg, split=partition)

        split_scores = score_df[score_df["_split"] == partition].copy()

        ds = PronunciationDataset(
            feature_arrays=arrays,
            meta_df=meta_df,
            cfg=cfg,
            hierarchical_scores=hierarchical_scores,
            score_df=split_scores,
        )

        X_list = []
        y_lists = {m: [] for m in active_metrics}
        
        for item in ds.samples:
            X_list.append(np.mean(item["pooled"], axis=0))
            for metric in active_metrics:
                tgt = item["targets"][metric]
                if isinstance(tgt, list):
                    # For phonemes and words, target is the mean score over the sentence
                    y_lists[metric].append(float(np.mean(tgt)) if tgt else 0.0)
                else:
                    y_lists[metric].append(float(tgt))

        X = np.stack(X_list)
        y = {m: np.array(vals) for m, vals in y_lists.items()}
        return X, y

    X_train, y_train = _pool_split("train")
    X_val, y_val = _pool_split("val")

    model_name = str(cfg.model.name)
    experiment_name = cfg.get("experiment_name", "apa_default")
    mlflow.set_experiment(experiment_name)

    models_dir = Path("models")
    models_dir.mkdir(exist_ok=True)

    # Train one model per active metric (baselines are single-output)
    for metric in active_metrics:
        log.info("Training %s baseline for metric '%s' …", model_name, metric)
        baseline = model_class(cfg)

        with mlflow.start_run(run_name=f"{model_name}_{metric}"):
            params = {
                "seed": int(cfg.seed),
                "git_hash": _get_git_hash(),
                "target_metric": metric,
            }
            if hasattr(baseline, "params_dict"):
                params.update(baseline.params_dict())
            mlflow.log_params(params)

            baseline.fit(X_train, y_train[metric])

            # Validation MSE
            y_pred_val = baseline.predict(X_val)
            val_mse = float(np.mean((y_pred_val - y_val[metric]) ** 2))
            mlflow.log_metric("val_mse", val_mse)
            log.info("  val_mse=%.4f", val_mse)

            save_path = models_dir / f"{model_name}_{metric}.pkl"
            baseline.save(save_path)
            mlflow.log_artifact(str(save_path), "models")


def _train_linear(cfg: DictConfig) -> None:
    """Train the Linear Regression baseline."""
    from src.models.linear_baseline import LinearBaseline
    _train_sklearn_baseline(cfg, LinearBaseline)


def _train_tree(cfg: DictConfig) -> None:
    """Train the XGBoost / Decision Tree baseline."""
    from src.models.tree_baseline import TreeBaseline
    _train_sklearn_baseline(cfg, TreeBaseline)


# ---------------------------------------------------------------------------
# Model dispatcher registry
# ---------------------------------------------------------------------------

_TRAINERS = {
    "bigru": _train_bigru,
    "linear": _train_linear,
    "tree": _train_tree,
}

# ---------------------------------------------------------------------------
# Hydra entrypoint
# ---------------------------------------------------------------------------


def _run_one_model(model_name: str, cfg: DictConfig) -> None:
    """Dispatch training for a single model by name."""
    if model_name not in _TRAINERS:
        raise ValueError(
            f"Unknown model '{model_name}'. "
            f"Valid choices: {list(_TRAINERS.keys())}."
        )
    log.info("=== Training model: %s ===", model_name)
    _TRAINERS[model_name](cfg)
    log.info("=== Finished model: %s ===", model_name)


def main() -> None:
    """
    CLI entrypoint — called when running ``python -m src.training.trainer``.

    Reads config via Hydra (configs/base.yaml + model override) and dispatches
    training.  The special model name ``"all"`` trains every model in sequence.
    """
    import hydra

    @hydra.main(
        version_base=None,
        config_path="../../configs",
        config_name="base",
    )
    def _hydra_main(cfg: DictConfig) -> None:
        logging.basicConfig(
            level=logging.INFO,
            format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
            datefmt="%H:%M:%S",
        )

        tracking_uri = cfg.get("mlflow_tracking_uri", "http://127.0.0.1:5000")
        mlflow_proc = _ensure_mlflow_server(tracking_uri)

        try:
            model_name: str = (
                str(cfg.model.name)
                if hasattr(cfg, "model") and hasattr(cfg.model, "name")
                else "all"
            )

            if model_name == "all":
                log.info("Training all models: %s", list(_TRAINERS.keys()))
                config_dir = Path(__file__).parent.parent.parent / "configs"
                # cfg is a Hydra struct — convert to a plain dict so we can
                # inject the 'model' key without triggering struct key errors.
                base_dict = OmegaConf.to_container(cfg, resolve=True, throw_on_missing=False)
                for name in _TRAINERS:
                    model_yaml = config_dir / "model" / f"{name}.yaml"
                    model_dict = OmegaConf.to_container(OmegaConf.load(model_yaml), resolve=True)
                    merged = dict(base_dict)        # shallow copy of base
                    merged["model"] = model_dict    # inject model sub-config
                    sub_cfg = OmegaConf.create(merged)
                    _run_one_model(name, sub_cfg)
            else:
                _run_one_model(model_name, cfg)
        finally:
            if mlflow_proc is not None:
                log.info("Shutting down MLflow server (PID %d).", mlflow_proc.pid)
                mlflow_proc.terminate()

    _hydra_main()


if __name__ == "__main__":
    main()
