"""
src/data/persist.py
===================
HDF5 serialization of extracted eGeMAPS feature arrays.

Responsibility
--------------
Serialize the raw feature DataFrame produced by ``extract.py`` to an HDF5
file stored under ``<features_dir>/features.h5``.

HDF5 layout
-----------
Each phoneme's feature vector is stored as a dataset named::

    /<speaker_id>/<sentence_id>/<phoneme_index>

Metadata (phoneme label, t_start, t_end) are attached as HDF5 attributes
on the dataset.

A top-level attribute ``split`` records which partition (train / val / test)
the file belongs to.

Usage
-----
    from src.data.persist import save_features, load_features

    # Saving
    save_features(feature_df, cfg, split="train")

    # Loading back into a DataFrame
    df = load_features(cfg, split="train")

Configuration keys
------------------
    features_dir : str
    seed         : int  (logged as attribute for reproducibility)
"""

import logging
from pathlib import Path

import h5py
import numpy as np
import pandas as pd
from omegaconf import DictConfig

log = logging.getLogger(__name__)

_FEATURE_COLS_PREFIX = "feat_"
_META_ATTRS = ("phoneme", "t_start", "t_end")


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------

def _feature_cols(df: pd.DataFrame) -> list[str]:
    """Return sorted list of ``feat_*`` column names present in ``df``."""
    return sorted(
        [c for c in df.columns if c.startswith(_FEATURE_COLS_PREFIX)],
        key=lambda c: int(c.split("_", 1)[1]),
    )


def _hdf5_path(features_dir: Path, split: str) -> Path:
    return features_dir / f"{split}_features.h5"


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def save_features(
    feature_df: pd.DataFrame,
    cfg: DictConfig,
    split: str,
) -> Path:
    """
    Persist a feature DataFrame to HDF5.

    Parameters
    ----------
    feature_df : pd.DataFrame
        Output of ``extract.extract_features()``.
        Required columns: speaker_id, sentence_id, phoneme_index,
        phoneme, t_start, t_end, feat_0 … feat_N.
    cfg : DictConfig
        Must contain: ``features_dir``, ``seed``.
    split : str
        One of ``"train"``, ``"val"``, ``"test"``.

    Returns
    -------
    Path
        Path to the written HDF5 file.
    """
    features_dir = Path(cfg.features_dir)
    features_dir.mkdir(parents=True, exist_ok=True)
    out_path = _hdf5_path(features_dir, split)

    feat_cols = _feature_cols(feature_df)
    if not feat_cols:
        raise ValueError("No feat_* columns found in feature_df.")

    n_features = len(feat_cols)

    with h5py.File(out_path, "w") as h5:
        # Top-level metadata
        h5.attrs["split"] = split
        h5.attrs["n_features"] = n_features
        h5.attrs["seed"] = int(cfg.seed)

        for _, row in feature_df.iterrows():
            spk = str(row["speaker_id"])
            sent = str(row["sentence_id"])
            ph_idx = int(row["phoneme_index"])

            ds_path = f"{spk}/{sent}/{ph_idx}"
            vector = row[feat_cols].to_numpy(dtype=np.float32)
            ds = h5.require_dataset(
                ds_path,
                shape=(n_features,),
                dtype=np.float32,
                data=vector,
            )
            # Attach metadata as attributes
            ds.attrs["phoneme"] = str(row["phoneme"])
            ds.attrs["t_start"] = float(row["t_start"])
            ds.attrs["t_end"] = float(row["t_end"])

    n_rows = len(feature_df)
    log.info(
        "Saved %d phoneme feature vectors → %s  (split='%s')",
        n_rows,
        out_path,
        split,
    )
    return out_path


def load_features(cfg: DictConfig, split: str) -> pd.DataFrame:
    """
    Load a previously saved HDF5 feature file back into a DataFrame.

    Parameters
    ----------
    cfg : DictConfig
        Must contain: ``features_dir``.
    split : str
        One of ``"train"``, ``"val"``, ``"test"``.

    Returns
    -------
    pd.DataFrame with columns:
        speaker_id, sentence_id, phoneme_index,
        phoneme, t_start, t_end, feat_0 … feat_N
    """
    features_dir = Path(cfg.features_dir)
    h5_path = _hdf5_path(features_dir, split)

    if not h5_path.exists():
        raise FileNotFoundError(
            f"HDF5 feature file not found: {h5_path}. "
            "Run the extraction pipeline first."
        )

    records: list[dict] = []

    with h5py.File(h5_path, "r") as h5:
        n_features: int = int(h5.attrs.get("n_features", 88))
        feat_cols = [f"feat_{i}" for i in range(n_features)]

        for spk in h5:
            for sent in h5[spk]:
                for ph_idx_str in h5[spk][sent]:
                    ds = h5[spk][sent][ph_idx_str]
                    vector: np.ndarray = ds[()]
                    record: dict = {
                        "speaker_id": spk,
                        "sentence_id": sent,
                        "phoneme_index": int(ph_idx_str),
                        "phoneme": ds.attrs.get("phoneme", ""),
                        "t_start": float(ds.attrs.get("t_start", 0.0)),
                        "t_end": float(ds.attrs.get("t_end", 0.0)),
                    }
                    for fi, fv in enumerate(vector):
                        record[feat_cols[fi]] = float(fv)
                    records.append(record)

    if not records:
        raise RuntimeError(f"No records loaded from {h5_path}.")

    df = pd.DataFrame(records).sort_values(
        ["speaker_id", "sentence_id", "phoneme_index"]
    ).reset_index(drop=True)

    log.info(
        "Loaded %d phoneme feature vectors from %s (split='%s')",
        len(df),
        h5_path,
        split,
    )
    return df


def list_splits(cfg: DictConfig) -> list[str]:
    """Return which splits have been persisted to HDF5."""
    features_dir = Path(cfg.features_dir)
    return [
        split
        for split in ("train", "val", "test")
        if _hdf5_path(features_dir, split).exists()
    ]
