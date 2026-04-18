"""
src/data/normalize.py
=====================
Standard scaling of eGeMAPS feature vectors.

Rules (from project spec)
-------------------------
1. Fit the ``StandardScaler`` **only** on the training partition.
2. Apply the fitted scaler to validation and test partitions.
3. **Never modify the scaler object after fitting** — transform calls are
   pure read-only operations on the already-fitted instance.
4. Persist the fitted scaler via ``joblib`` so evaluation scripts can reuse
   the exact same transformation artefact.

Usage
-----
    from src.data.normalize import fit_scaler, transform_split, load_scaler

    # During preprocessing (training time only):
    scaler = fit_scaler(train_feature_df, cfg)
    train_scaled = transform_split(train_feature_df, scaler)
    val_scaled   = transform_split(val_feature_df, scaler)
    test_scaled  = transform_split(test_feature_df, scaler)

    # During evaluation / inference:
    scaler = load_scaler(cfg)
    test_scaled = transform_split(test_feature_df, scaler)

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
_FEATURE_COLS_PREFIX = "feat_"


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------

def _feature_cols(df: pd.DataFrame) -> list[str]:
    """Return sorted list of ``feat_*`` column names present in ``df``."""
    return sorted(
        [c for c in df.columns if c.startswith(_FEATURE_COLS_PREFIX)],
        key=lambda c: int(c.split("_", 1)[1]),
    )


def _scaler_path(cfg: DictConfig) -> Path:
    return Path(cfg.scalers_dir) / _SCALER_FILENAME


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def fit_scaler(train_df: pd.DataFrame, cfg: DictConfig) -> StandardScaler:
    """
    Fit a ``StandardScaler`` on the training partition's feature columns.

    The scaler is immediately saved to disk (``scalers_dir/standard_scaler.joblib``).

    Parameters
    ----------
    train_df : pd.DataFrame
        Training-partition feature DataFrame (output of ``persist.load_features``).
        Must contain ``feat_0 … feat_N`` columns.
    cfg : DictConfig
        Must contain: ``scalers_dir``, ``seed`` (logged only).

    Returns
    -------
    StandardScaler
        A fully fitted scaler. **Do not call .fit() or .fit_transform() on this
        object again** — it must remain frozen after this call.
    """
    scalers_dir = Path(cfg.scalers_dir)
    scalers_dir.mkdir(parents=True, exist_ok=True)

    feat_cols = _feature_cols(train_df)
    if not feat_cols:
        raise ValueError(
            "No feat_* columns found in train_df. "
            "Ensure extraction ran before normalization."
        )

    X_train: np.ndarray = train_df[feat_cols].to_numpy(dtype=np.float64)

    # ── Fit (training partition ONLY) ────────────────────────────────────────
    scaler = StandardScaler()
    scaler.fit(X_train)

    # ── Persist immediately — never modify after this point ──────────────────
    out_path = _scaler_path(cfg)
    joblib.dump(scaler, out_path)

    log.info(
        "StandardScaler fitted on %d training samples (%d features). "
        "Train mean range: [%.4f, %.4f]. Saved → %s",
        len(X_train),
        len(feat_cols),
        float(scaler.mean_.min()),
        float(scaler.mean_.max()),
        out_path,
    )

    return scaler


def transform_split(
    feature_df: pd.DataFrame,
    scaler: StandardScaler,
) -> pd.DataFrame:
    """
    Apply a **pre-fitted** scaler to a feature DataFrame (in-place replacement
    of feat_* columns).

    Parameters
    ----------
    feature_df : pd.DataFrame
        Any split (train / val / test) — the scaler must already be fitted.
    scaler : StandardScaler
        A fitted scaler. This function **only calls ``scaler.transform()``**
        — it never re-fits.

    Returns
    -------
    pd.DataFrame
        Copy of ``feature_df`` with ``feat_*`` columns replaced by scaled values.

    Raises
    ------
    RuntimeError
        If the scaler has not been fitted yet.
    """
    if not hasattr(scaler, "mean_"):
        raise RuntimeError(
            "The provided scaler has not been fitted. "
            "Call fit_scaler(train_df, cfg) first."
        )

    feat_cols = _feature_cols(feature_df)
    if not feat_cols:
        raise ValueError("No feat_* columns found in feature_df.")

    X: np.ndarray = feature_df[feat_cols].to_numpy(dtype=np.float64)

    # Pure transform — no fitting
    X_scaled: np.ndarray = scaler.transform(X)

    result = feature_df.copy()
    result[feat_cols] = X_scaled.astype(np.float32)

    log.debug(
        "Transformed %d samples. Post-scale mean≈%.4f, std≈%.4f",
        len(X_scaled),
        float(X_scaled.mean()),
        float(X_scaled.std()),
    )
    return result


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
            "Run fit_scaler(train_df, cfg) during preprocessing."
        )

    scaler: StandardScaler = joblib.load(path)

    if not hasattr(scaler, "mean_"):
        raise RuntimeError(
            f"Loaded object from {path} is not a fitted StandardScaler."
        )

    log.info("Loaded fitted StandardScaler from %s", path)
    return scaler


def run_normalization(
    train_df: pd.DataFrame,
    val_df: pd.DataFrame,
    test_df: pd.DataFrame,
    cfg: DictConfig,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, StandardScaler]:
    """
    Convenience function: fit on train, transform all three splits.

    Returns
    -------
    (train_scaled, val_scaled, test_scaled, scaler)
    """
    scaler = fit_scaler(train_df, cfg)
    train_scaled = transform_split(train_df, scaler)
    val_scaled = transform_split(val_df, scaler)
    test_scaled = transform_split(test_df, scaler)
    return train_scaled, val_scaled, test_scaled, scaler
