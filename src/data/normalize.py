"""
src/data/normalize.py
=====================
Standard scaling of eGeMAPS LLD frame sequences.

Rules (from project spec)
-------------------------
1. Fit the ``StandardScaler`` **only** on the training partition.
2. Apply the fitted scaler to validation and test partitions.
3. **Never modify the scaler object after fitting** — transform calls are
   pure read-only operations on the already-fitted instance.
4. Persist the fitted scaler via ``joblib`` so evaluation scripts can reuse
   the exact same transformation artefact.

Frame-level normalization
--------------------------
Each phoneme is stored as a 2-D array ``(T_frames, 25)``.  To fit the scaler
we concatenate **all frames from all training-partition phonemes** into a
single ``(N_total_frames, 25)`` matrix and fit the ``StandardScaler`` over
the frame axis.

Applying the scaler is equally frame-level: for each phoneme array the
scaler transforms every frame independently.

Usage
-----
    from src.data.normalize import fit_scaler, transform_arrays, load_scaler

    # During preprocessing (training time only):
    scaler = fit_scaler(train_arrays, cfg)
    train_scaled = transform_arrays(train_arrays, scaler)
    val_scaled   = transform_arrays(val_arrays,   scaler)
    test_scaled  = transform_arrays(test_arrays,  scaler)

    # During evaluation / inference:
    scaler = load_scaler(cfg)
    test_scaled = transform_arrays(test_arrays, scaler)

Configuration keys
------------------
    scalers_dir : str   directory for the saved scaler artefact
    seed        : int   logged for reproducibility (scaler itself is deterministic)
"""

import logging
from pathlib import Path

import numpy as np
import pandas as pd
import joblib
from omegaconf import DictConfig
from sklearn.preprocessing import StandardScaler

log = logging.getLogger(__name__)

_SCALER_FILENAME = "standard_scaler.joblib"

# Type alias for the feature dict produced by extract / persist.
# Keys are either (speaker_id, sentence_id, phoneme_index) when phoneme
# boundaries are used, or (speaker_id, sentence_id) in the no-boundary case.
FeatureArrays = dict[tuple, np.ndarray]


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------

def _scaler_path(cfg: DictConfig) -> Path:
    return Path(cfg.scalers_dir) / _SCALER_FILENAME


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def fit_scaler(train_arrays: FeatureArrays, cfg: DictConfig) -> StandardScaler:
    """
    Fit a ``StandardScaler`` on all LLD frames in the training partition.

    All phoneme frame arrays are stacked into a single ``(N_frames, 23)``
    matrix before fitting, so normalization is computed over the per-feature
    global statistics across all training frames.

    The scaler is immediately saved to disk
    (``scalers_dir/standard_scaler.joblib``).

    Parameters
    ----------
    train_arrays : dict[(speaker_id, sentence_id, phoneme_index) -> np.ndarray]
        Training-partition feature arrays (output of ``persist.load_features``
        or ``extract.extract_features``).  Each value is shape
        ``(T_frames, 25)``.
    cfg : DictConfig
        Must contain: ``scalers_dir``, ``seed`` (logged only).

    Returns
    -------
    StandardScaler
        A fully fitted scaler. **Do not call .fit() or .fit_transform() on
        this object again** — it must remain frozen after this call.
    """
    scalers_dir = Path(cfg.scalers_dir)
    scalers_dir.mkdir(parents=True, exist_ok=True)

    if not train_arrays:
        raise ValueError(
            "train_arrays is empty. "
            "Ensure extraction ran before normalization."
        )

    # Stack all frames: (N_total_frames, 23)
    all_frames = np.vstack(list(train_arrays.values())).astype(np.float64)

    if all_frames.ndim != 2:
        raise ValueError(
            f"Expected 2-D frame matrix, got shape {all_frames.shape}."
        )

    n_frames, n_features = all_frames.shape
    log.info(
        "Fitting StandardScaler on %d training frames (%d features) "
        "from %d sequences.",
        n_frames,
        n_features,
        len(train_arrays),
    )

    # ── Fit (training frames ONLY) ───────────────────────────────────────────
    scaler = StandardScaler()
    scaler.fit(all_frames)

    # ── Persist immediately — never modify after this point ──────────────────
    out_path = _scaler_path(cfg)
    joblib.dump(scaler, out_path)

    log.info(
        "StandardScaler fitted. Mean range: [%.4f, %.4f]. Saved → %s",
        float(scaler.mean_.min()),
        float(scaler.mean_.max()),
        out_path,
    )

    return scaler


def transform_arrays(
    feature_arrays: FeatureArrays,
    scaler: StandardScaler,
) -> FeatureArrays:
    """
    Apply a **pre-fitted** scaler to a set of LLD frame arrays.

    Each phoneme array ``(T_frames, 25)`` is transformed independently —
    the scaler is applied frame-by-frame (i.e., ``scaler.transform(arr)``
    where ``arr`` has shape ``(T_frames, 25)``).

    Parameters
    ----------
    feature_arrays : dict[(speaker_id, sentence_id, phoneme_index) -> np.ndarray]
        Any split (train / val / test) — the scaler must already be fitted.
    scaler : StandardScaler
        A fitted scaler. This function **only calls ``scaler.transform()``**
        — it never re-fits.

    Returns
    -------
    dict[(speaker_id, sentence_id, phoneme_index) -> np.ndarray]
        New dict with scaled arrays of the same shapes as the input.

    Raises
    ------
    RuntimeError
        If the scaler has not been fitted yet.
    """
    if not hasattr(scaler, "mean_"):
        raise RuntimeError(
            "The provided scaler has not been fitted. "
            "Call fit_scaler(train_arrays, cfg) first."
        )

    scaled: FeatureArrays = {}
    for key, frames in feature_arrays.items():
        # frames: (T_frames, 23)
        frames_f64 = frames.astype(np.float64)
        scaled_frames = scaler.transform(frames_f64).astype(np.float32)
        scaled[key] = scaled_frames

    # Quick summary stat for the log
    if scaled:
        sample_vals = np.vstack(list(scaled.values()))
        log.debug(
            "Transformed %d phoneme sequences. "
            "Post-scale mean≈%.4f, std≈%.4f",
            len(scaled),
            float(sample_vals.mean()),
            float(sample_vals.std()),
        )

    return scaled


def load_scaler(cfg: DictConfig) -> StandardScaler:
    """
    Load the previously fitted scaler from disk.

    Parameters
    ----------
    cfg : DictConfig
        Must contain: ``scalers_dir``.

    Returns
    -------
    StandardScaler
        Fitted scaler ready for ``transform()`` calls only.

    Raises
    ------
    FileNotFoundError
        If the scaler artefact does not exist.
    """
    path = _scaler_path(cfg)
    if not path.exists():
        raise FileNotFoundError(
            f"Scaler not found at {path}. "
            "Run fit_scaler(train_arrays, cfg) during preprocessing."
        )

    scaler: StandardScaler = joblib.load(path)

    if not hasattr(scaler, "mean_"):
        raise RuntimeError(
            f"Loaded object from {path} is not a fitted StandardScaler."
        )

    log.info("Loaded fitted StandardScaler from %s", path)
    return scaler


def run_normalization(
    train_arrays: FeatureArrays,
    val_arrays: FeatureArrays,
    test_arrays: FeatureArrays,
    cfg: DictConfig,
) -> tuple[FeatureArrays, FeatureArrays, FeatureArrays, StandardScaler]:
    """
    Convenience function: fit on train frames, transform all three splits.

    Returns
    -------
    (train_scaled, val_scaled, test_scaled, scaler)
    """
    scaler = fit_scaler(train_arrays, cfg)
    train_scaled = transform_arrays(train_arrays, scaler)
    val_scaled = transform_arrays(val_arrays, scaler)
    test_scaled = transform_arrays(test_arrays, scaler)
    return train_scaled, val_scaled, test_scaled, scaler


import hydra


@hydra.main(version_base=None, config_path="../../configs", config_name="base")
def main(cfg: DictConfig) -> None:
    from src.data.persist import load_features, save_features, load_meta

    splits: dict[str, FeatureArrays] = {}
    metas: dict[str, pd.DataFrame] = {}
    for partition in ("train", "val", "test"):
        arrays, meta = load_features(cfg, split=partition)
        splits[partition] = arrays
        metas[partition] = meta

    train_scaled, val_scaled, test_scaled, _ = run_normalization(
        splits["train"], splits["val"], splits["test"], cfg,
    )

    for partition, arrays in [
        ("train", train_scaled),
        ("val", val_scaled),
        ("test", test_scaled),
    ]:
        save_features(arrays, metas[partition], cfg, split=f"{partition}_scaled")

    log.info("Normalization complete. Scaler saved to %s", _scaler_path(cfg))


if __name__ == "__main__":
    main()
