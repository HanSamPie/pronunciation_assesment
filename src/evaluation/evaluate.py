"""
src/evaluation/evaluate.py
==========================
Evaluation orchestration — runs inference for all models on all splits,
caches results, and generates charts.

CLI usage::

    make eval
    python -m src.evaluation.evaluate

The script:
1. Loads config (base.yaml + eval.yaml)
2. For each model × split combination:
   a. Checks the SQLite cache for existing results
   b. If cache miss: runs inference, computes metrics, caches results
3. Generates per-split charts in ``outputs/evaluation/{split}/``
"""

from __future__ import annotations

import logging
import sys
from pathlib import Path
from typing import Dict, Optional, Tuple

import numpy as np
import pandas as pd
import torch
from omegaconf import DictConfig, OmegaConf
from torch.utils.data import DataLoader

from src.evaluation.cache import EvaluationCache, generate_cache_key, hash_file

log = logging.getLogger(__name__)

# Metrics computation (original utility functions preserved)
from scipy.stats import pearsonr, spearmanr
from sklearn.metrics import mean_squared_error


def compute_metrics(y_true: np.ndarray, y_pred: np.ndarray) -> Dict[str, float]:
    """
    Computes RMSE, PCC, and SRC for a given set of targets and predictions.
    """
    y_true = np.asarray(y_true, dtype=float)
    y_pred = np.asarray(y_pred, dtype=float)

    rmse = np.sqrt(mean_squared_error(y_true, y_pred))

    # PCC and SRC can be undefined if variance is 0 (all predictions or targets are the same)
    if len(np.unique(y_true)) > 1 and len(np.unique(y_pred)) > 1:
        pcc, _ = pearsonr(y_true, y_pred)
        src, _ = spearmanr(y_true, y_pred)
    else:
        pcc = float('nan')
        src = float('nan')

    return {
        "rmse": float(rmse),
        "pcc": float(pcc),
        "src": float(src)
    }


def evaluate_all_metrics(
    targets: Dict[str, np.ndarray], predictions: Dict[str, np.ndarray]
) -> Dict[str, Dict[str, float]]:
    """
    Evaluates all metrics present in both targets and predictions.

    Args:
        targets: Dictionary mapping metric name to 1D array of ground truth scores.
        predictions: Dictionary mapping metric name to 1D array of predicted scores.

    Returns:
        Dictionary mapping metric name to a dictionary of (rmse, pcc, src).
    """
    results = {}
    for metric_name in targets.keys():
        if metric_name in predictions:
            results[metric_name] = compute_metrics(
                targets[metric_name], predictions[metric_name]
            )
    return results


# ---------------------------------------------------------------------------
# Data loading helpers
# ---------------------------------------------------------------------------

_SENTENCE_KEY_MAP: dict[str, str] = {
    "sentence_accuracy":     "accuracy",
    "sentence_fluency":      "fluency",
    "sentence_prosodic":     "prosodic",
}
_WORD_KEY_MAP: dict[str, str] = {
    "word_accuracy": "accuracy",
    "word_stress":   "stress",
}


def _load_split_data(cfg: DictConfig, split: str):
    """
    Load features, manifest, and scores for a single split.

    Returns
    -------
    arrays : dict
        Feature arrays keyed by (spk, sent, ph_idx).
    manifest_df : pd.DataFrame
        Manifest for this split with speaker_id, sentence_id, age, gender, etc.
    hierarchical_scores : dict
        ``scores.json`` keyed by 9-digit sentence ID.
    """
    from src.data.persist import load_features
    from src.data.scores import load_hierarchical_scores

    # Load features
    try:
        arrays, meta_df = load_features(cfg, split=f"{split}_scaled")
    except FileNotFoundError:
        log.warning(
            "Scaled features not found for '%s'. Trying un-scaled features …",
            split,
        )
        arrays, meta_df = load_features(cfg, split=split)

    # Load manifest
    splits_dir = Path(cfg.splits_dir)
    csv_path = splits_dir / f"{split}_manifest.csv"
    if not csv_path.exists():
        raise FileNotFoundError(f"Manifest not found: {csv_path}")
    manifest_df = pd.read_csv(csv_path)
    manifest_df["speaker_id"] = manifest_df["speaker_id"].astype(str).str.zfill(4)
    manifest_df["sentence_id"] = manifest_df["sentence_id"].astype(str).str.zfill(9)

    # Load hierarchical scores
    scores_path = Path(cfg.scores_file)
    hierarchical_scores = load_hierarchical_scores(scores_path)

    return arrays, manifest_df, hierarchical_scores


def _pool_sentences(
    arrays: dict,
    manifest_df: pd.DataFrame,
    hierarchical_scores: dict,
    active_metrics: list[str],
) -> Tuple[np.ndarray, Dict[str, np.ndarray], list[str]]:
    """
    Mean-pool features per sentence and extract ground truth targets.

    Used for baseline (linear/tree) inference.

    Returns
    -------
    X : np.ndarray
        (N_sentences, n_features) pooled feature matrix.
    targets : dict
        {metric_name: np.ndarray of shape (N_sentences,)}
    speaker_ids : list[str]
        Speaker ID for each sentence (for fairness analysis).
    """
    valid_sents = set(
        zip(
            manifest_df["speaker_id"].astype(str),
            manifest_df["sentence_id"].astype(str),
        )
    )

    # Group frames by sentence
    sentences: dict[tuple, list] = {}
    for (spk, sent, ph_idx), arr in arrays.items():
        key = (spk, sent)
        if key not in sentences:
            sentences[key] = []
        sentences[key].append(arr.mean(axis=0))

    X_list = []
    y_lists: dict[str, list] = {m: [] for m in active_metrics}
    speaker_ids: list[str] = []

    for (spk, sent), pooled_list in sentences.items():
        if (str(spk), str(sent)) not in valid_sents:
            continue
        entry = hierarchical_scores.get(str(sent).zfill(9))
        if entry is None:
            continue

        x = np.mean(pooled_list, axis=0)
        X_list.append(x)
        speaker_ids.append(str(spk))

        for metric in active_metrics:
            if metric.startswith("sentence_"):
                json_key = _SENTENCE_KEY_MAP.get(metric, "accuracy")
            elif metric.startswith("word_"):
                json_key = _WORD_KEY_MAP.get(metric, "accuracy")
            else:
                json_key = "accuracy"
            y_lists[metric].append(float(entry.get(json_key, 0.0)))

    X = np.stack(X_list)
    targets = {m: np.array(vals) for m, vals in y_lists.items()}
    return X, targets, speaker_ids


# ---------------------------------------------------------------------------
# Inference: BiGRU
# ---------------------------------------------------------------------------


def run_bigru_inference(
    cfg: DictConfig, checkpoint_path: Path, split: str
) -> Tuple[Dict[str, np.ndarray], Dict[str, np.ndarray], list[str]]:
    """
    Load a BiGRU checkpoint, run inference on the given split, return
    per-metric targets and predictions.

    Returns
    -------
    targets : dict[str, np.ndarray]
    predictions : dict[str, np.ndarray]
    speaker_ids : list[str]
    """
    from src.models.bigru import HierarchicalBiGRU
    from src.training.trainer import PronunciationDataset, _collate_fn, _load_scores
    from src.data.persist import load_features

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    log.info("BiGRU inference on '%s' split (device=%s, ckpt=%s)", split, device, checkpoint_path)

    # Inject bigru model config if not already present
    if not hasattr(cfg, "model") or cfg.get("model") is None:
        config_dir = Path(__file__).parent.parent.parent / "configs"
        bigru_yaml = config_dir / "model" / "bigru.yaml"
        if bigru_yaml.exists():
            model_dict = OmegaConf.to_container(OmegaConf.load(bigru_yaml), resolve=True)
            base_dict = OmegaConf.to_container(cfg, resolve=True) if not isinstance(cfg, dict) else dict(cfg)
            base_dict["model"] = model_dict
            cfg = OmegaConf.create(base_dict)

    # Ensure dropout and boundary jitter are disabled during inference
    from omegaconf import open_dict
    with open_dict(cfg):
        cfg.use_boundary_jitter = False
        if hasattr(cfg, "model") and cfg.model is not None:
            cfg.model.dropout = 0.0

    # Load features
    try:
        arrays, meta_df = load_features(cfg, split=f"{split}_scaled")
    except FileNotFoundError:
        arrays, meta_df = load_features(cfg, split=split)

    score_df, hierarchical_scores = _load_scores(cfg)
    split_scores = score_df[score_df["_split"] == split].copy()

    ds = PronunciationDataset(
        feature_arrays=arrays,
        meta_df=meta_df,
        cfg=cfg,
        hierarchical_scores=hierarchical_scores,
        score_df=split_scores,
    )

    loader = DataLoader(
        ds, batch_size=64, shuffle=False, collate_fn=_collate_fn, num_workers=0
    )

    # Infer input size from data
    sample_batch = next(iter(loader))
    n_lld = sample_batch["phoneme_features"].shape[-1]

    # Build model and load checkpoint
    model = HierarchicalBiGRU(cfg, input_size=n_lld)
    state_dict = torch.load(checkpoint_path, map_location=device, weights_only=True)
    model.load_state_dict(state_dict)
    model.to(device)
    model.eval()

    # Collect predictions and targets
    all_preds: dict[str, list] = {}
    all_targets: dict[str, list] = {}

    with torch.no_grad():
        for batch in loader:
            features = batch["phoneme_features"].to(device)
            mask = batch["phoneme_mask"].to(device)
            word_bounds = batch["word_boundaries"]
            targets = batch["targets"]

            preds = model(features, mask, word_bounds)

            for metric in ds.active_metrics:
                if metric not in preds or metric not in targets:
                    continue
                pred_np = preds[metric].cpu().numpy()
                tgt_np = targets[metric].cpu().numpy() if isinstance(targets[metric], torch.Tensor) else targets[metric]

                all_preds.setdefault(metric, []).append(pred_np)
                all_targets.setdefault(metric, []).append(
                    tgt_np if isinstance(tgt_np, np.ndarray) else np.array(tgt_np)
                )

    # Flatten
    flat_preds = {m: np.concatenate(v) for m, v in all_preds.items()}
    flat_targets = {m: np.concatenate(v) for m, v in all_targets.items()}

    # Collect speaker IDs from the dataset samples
    speaker_ids = [str(s["spk"]) for s in ds.samples]

    # For sentence-level metrics, speaker_ids is 1:1.
    # For phoneme/word-level, we need to expand speaker IDs to match.
    # We'll store sentence-level speaker_ids and let charts handle expansion.

    return flat_targets, flat_preds, speaker_ids


# ---------------------------------------------------------------------------
# Inference: Baselines (Linear / Tree)
# ---------------------------------------------------------------------------


def run_baseline_inference(
    cfg: DictConfig, model_type: str, split: str
) -> Tuple[Dict[str, np.ndarray], Dict[str, np.ndarray], list[str]]:
    """
    Load per-metric baseline models and run inference on the given split.

    Returns
    -------
    targets : dict[str, np.ndarray]
    predictions : dict[str, np.ndarray]
    speaker_ids : list[str]
    """
    from src.models.linear_baseline import LinearBaseline
    from src.models.tree_baseline import TreeBaseline
    from src.training.trainer import PronunciationDataset, _load_scores
    from src.data.persist import load_features

    # Inject model config if not already present
    if not hasattr(cfg, "model") or cfg.get("model") is None:
        config_dir = Path(__file__).parent.parent.parent / "configs"
        model_yaml = config_dir / "model" / f"{model_type}.yaml"
        if model_yaml.exists():
            model_dict = OmegaConf.to_container(OmegaConf.load(model_yaml), resolve=True)
            base_dict = OmegaConf.to_container(cfg, resolve=True) if not isinstance(cfg, dict) else dict(cfg)
            base_dict["model"] = model_dict
            cfg = OmegaConf.create(base_dict)

    # Ensure boundary jitter is disabled during inference
    from omegaconf import open_dict
    with open_dict(cfg):
        cfg.use_boundary_jitter = False

    active_metrics = list(
        OmegaConf.to_container(cfg.metrics[cfg.score_mode], resolve=True).keys()
    )

    try:
        arrays, meta_df = load_features(cfg, split=f"{split}_scaled")
    except FileNotFoundError:
        arrays, meta_df = load_features(cfg, split=split)

    score_df, hierarchical_scores = _load_scores(cfg)
    split_scores = score_df[score_df["_split"] == split].copy()

    ds = PronunciationDataset(
        feature_arrays=arrays,
        meta_df=meta_df,
        cfg=cfg,
        hierarchical_scores=hierarchical_scores,
        score_df=split_scores,
    )

    X_list = []
    targets_lists: dict[str, list] = {m: [] for m in active_metrics}
    speaker_ids = []

    for item in ds.samples:
        X_list.append(np.mean(item["pooled"], axis=0))
        speaker_ids.append(str(item["spk"]))
        for m in active_metrics:
            targets_lists[m].append(item["targets"][m])

    X = np.stack(X_list)
    models_dir = Path("models")
    predictions: dict[str, np.ndarray] = {}
    targets_flat: dict[str, np.ndarray] = {}

    for metric in active_metrics:
        model_path = models_dir / f"{model_type}_{metric}.pkl"
        if not model_path.exists():
            log.warning("Baseline model not found: %s — skipping %s", model_path, metric)
            continue

        if model_type == "linear":
            model = LinearBaseline.load(model_path, cfg=cfg)
        elif model_type == "tree":
            model = TreeBaseline.load(model_path, cfg=cfg)
        else:
            raise ValueError(f"Unknown baseline type: {model_type}")

        preds_sentence = model.predict(X)
        
        preds_list = []
        for i, item in enumerate(ds.samples):
            pred_val = preds_sentence[i]
            tgt = item["targets"][metric]
            
            if isinstance(tgt, list):
                # Broadcast the single sentence prediction to all words/phonemes
                preds_list.extend([pred_val] * len(tgt))
            else:
                preds_list.append(pred_val)

        predictions[metric] = np.array(preds_list)

        # Flatten targets
        if isinstance(targets_lists[metric][0], list):
            tgt_flat = []
            for t in targets_lists[metric]:
                tgt_flat.extend(t)
            targets_flat[metric] = np.array(tgt_flat)
        else:
            targets_flat[metric] = np.array(targets_lists[metric])

        log.info("  %s → %s: predicted %d samples", model_type, metric, len(predictions[metric]))

    return targets_flat, predictions, speaker_ids


# ---------------------------------------------------------------------------
# Single model/split evaluation
# ---------------------------------------------------------------------------


def evaluate_model_on_split(
    cfg: DictConfig,
    model_name: str,
    split: str,
    cache: EvaluationCache,
    checkpoint_path: Optional[Path] = None,
) -> Tuple[Dict[str, Dict[str, float]], Dict[str, np.ndarray], Dict[str, np.ndarray], list[str]]:
    """
    Evaluate a single model on a single split.  Uses cache if available.

    Returns
    -------
    results : dict
        {metric_name: {rmse, pcc, src}}
    targets : dict
    predictions : dict
    speaker_ids : list[str]
    """
    # Determine model weights path for cache key
    if checkpoint_path is not None:
        weights_path = checkpoint_path
    else:
        # For baselines, hash all per-metric model files
        models_dir = Path("models")
        metric_files = sorted(models_dir.glob(f"{model_name}_*.pkl"))
        if metric_files:
            weights_path = metric_files[0]  # use first for cache key
        else:
            weights_path = Path("no_model")

    # Data manifest for cache key
    data_manifest_path = Path(cfg.splits_dir) / f"{split}_manifest.csv"

    cache_key = generate_cache_key(
        cfg, weights_path, data_manifest_path, model_name, split
    )

    # Check cache
    cached = cache.get(cache_key)
    if cached is not None:
        log.info("Using cached results for %s / %s", model_name, split)
        # We still need targets/predictions for charts — run inference
        # but return cached metrics
        targets, predictions, speaker_ids = _run_inference(
            cfg, model_name, split, checkpoint_path
        )
        return cached, targets, predictions, speaker_ids

    # Cache miss — run inference
    log.info("Evaluating %s on %s split …", model_name, split)
    targets, predictions, speaker_ids = _run_inference(
        cfg, model_name, split, checkpoint_path
    )

    # Compute metrics
    results = evaluate_all_metrics(targets, predictions)

    # Cache the results
    cache.put(cache_key, results, model_name, split)

    return results, targets, predictions, speaker_ids


def _run_inference(
    cfg: DictConfig,
    model_name: str,
    split: str,
    checkpoint_path: Optional[Path],
) -> Tuple[Dict[str, np.ndarray], Dict[str, np.ndarray], list[str]]:
    """Dispatch inference to the correct model type."""
    if model_name.startswith("bigru"):
        if checkpoint_path is None:
            raise ValueError("checkpoint_path required for BiGRU inference")
        return run_bigru_inference(cfg, checkpoint_path, split)
    elif model_name in ("linear", "tree"):
        return run_baseline_inference(cfg, model_name, split)
    else:
        raise ValueError(f"Unknown model: {model_name}")


# ---------------------------------------------------------------------------
# Main orchestration
# ---------------------------------------------------------------------------


def main() -> None:
    """
    CLI entrypoint — runs evaluation for all models × all splits.
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

        # Merge eval config — convert Hydra struct to plain dict first
        eval_yaml = Path(__file__).parent.parent.parent / "configs" / "eval.yaml"
        if eval_yaml.exists():
            eval_cfg = OmegaConf.load(eval_yaml)
            base_dict = OmegaConf.to_container(cfg, resolve=True)
            eval_dict = OmegaConf.to_container(eval_cfg, resolve=True)
            base_dict.update(eval_dict)
            cfg = OmegaConf.create(base_dict)

        eval_output_dir = Path(cfg.get("eval_output_dir", "outputs/evaluation"))
        cache_db = str(cfg.get("eval_cache_db", "results.db"))
        cache = EvaluationCache(db_path=cache_db)

        # Determine models to evaluate
        bigru_checkpoints = list(cfg.get("bigru_checkpoints", []))
        baseline_models = []
        models_dir = Path("models")
        for btype in ("linear", "tree"):
            if list(models_dir.glob(f"{btype}_*.pkl")):
                baseline_models.append(btype)

        splits = ["train", "val", "test"]

        # ── Collect all results for cross-model charts ──────────────────
        # Structure: {split: {model_name: {metric: {rmse, pcc, src}}}}
        all_results: dict[str, dict[str, dict]] = {s: {} for s in splits}
        # Structure: {split: {model_name: (targets, predictions, speaker_ids)}}
        all_data: dict[str, dict[str, tuple]] = {s: {} for s in splits}

        # Load manifests for demographic charts
        manifests: dict[str, pd.DataFrame] = {}
        for split in splits:
            csv_path = Path(cfg.splits_dir) / f"{split}_manifest.csv"
            if csv_path.exists():
                df = pd.read_csv(csv_path)
                df["speaker_id"] = df["speaker_id"].astype(str).str.zfill(4)
                df["sentence_id"] = df["sentence_id"].astype(str).str.zfill(9)
                manifests[split] = df

        print("═" * 60)
        print("  Evaluation Pipeline")
        print("═" * 60)

        # ── Evaluate BiGRU checkpoints ──────────────────────────────────
        for ckpt_path_str in bigru_checkpoints:
            ckpt_path = Path(ckpt_path_str)
            if not ckpt_path.exists():
                log.warning("BiGRU checkpoint not found: %s — skipping", ckpt_path)
                continue

            # Use short hash from filename for model name
            ckpt_stem = ckpt_path.stem  # e.g. bigru_85abac2d...
            model_name = ckpt_stem  # full name for identifiability

            for split in splits:
                print(f"\n▸ {model_name} / {split}")
                results, targets, predictions, speaker_ids = evaluate_model_on_split(
                    cfg, model_name, split, cache, checkpoint_path=ckpt_path
                )
                all_results[split][model_name] = results
                all_data[split][model_name] = (targets, predictions, speaker_ids)
                _print_results(model_name, split, results)

        # ── Evaluate baselines ──────────────────────────────────────────
        for model_name in baseline_models:
            for split in splits:
                print(f"\n▸ {model_name} / {split}")
                results, targets, predictions, speaker_ids = evaluate_model_on_split(
                    cfg, model_name, split, cache
                )
                all_results[split][model_name] = results
                all_data[split][model_name] = (targets, predictions, speaker_ids)
                _print_results(model_name, split, results)

        # ── Generate charts ─────────────────────────────────────────────
        print("\n" + "═" * 60)
        print("  Generating Charts")
        print("═" * 60)

        from src.evaluation.charts import generate_all_charts

        active_metrics = list(
            OmegaConf.to_container(cfg.metrics[cfg.score_mode], resolve=True).keys()
        )

        generate_all_charts(
            all_results=all_results,
            all_data=all_data,
            manifests=manifests,
            active_metrics=active_metrics,
            bigru_checkpoints=bigru_checkpoints,
            output_dir=eval_output_dir,
        )

        # ── Summary ─────────────────────────────────────────────────────
        print("\n" + "═" * 60)
        print("  Cache contents")
        print("═" * 60)
        for model, split, created_at in cache.list_cached():
            print(f"  {model:45s}  {split:6s}  {created_at}")

        print("\n" + "═" * 60)
        print(f"  Charts saved to: {eval_output_dir}/")
        print(f"  Cache database:  {cache_db}")
        print("═" * 60)

    _hydra_main()


def _print_results(model_name: str, split: str, results: dict) -> None:
    """Pretty-print evaluation results."""
    for metric, scores in results.items():
        pcc = scores.get("pcc", float("nan"))
        rmse = scores.get("rmse", float("nan"))
        src = scores.get("src", float("nan"))
        print(f"    {metric:30s}  PCC={pcc:.4f}  RMSE={rmse:.4f}  SRC={src:.4f}")


if __name__ == "__main__":
    main()
