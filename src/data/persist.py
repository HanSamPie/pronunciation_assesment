"""
src/data/persist.py
===================
HDF5 serialization of extracted eGeMAPS LLD frame sequences.

Responsibility
--------------
Serialize the variable-length feature arrays produced by ``extract.py`` to
an HDF5 file stored under ``<features_dir>/<split>_features.h5``.

Each phoneme maps to a 2-D array of shape ``(T_frames, 23)`` where
``T_frames`` is variable per phoneme (depends on duration and openSMILE
frame-step).  This replaces the previous fixed-size 88-dim per-phoneme
vector storage.

HDF5 layout
-----------
::

    /  (root)
    ├── attrs:
    │   ├── split      (str)   partition label
    │   ├── n_features (int)   LLD dimensionality, always 23
    │   └── seed       (int)   RNG seed for reproducibility
    │
    └── <speaker_id>/
        └── <sentence_id>/
            └── <phoneme_index>/    ← dataset, shape (T_frames, 23)
                ├── attrs:
                │   ├── phoneme  (str)
                │   ├── t_start  (float)
                │   ├── t_end    (float)
                │   └── n_frames (int)

A companion lightweight **metadata table** is stored as a Pandas DataFrame
pickled inside the HDF5 under the ``/meta`` dataset (as a variable-length
byte string) so callers can inspect the index without loading all arrays.

Usage
-----
    from src.data.persist import save_features, load_features, load_meta

    # Saving (output of extract.extract_features)
    save_features(feature_arrays, meta_df, cfg, split="train")

    # Loading metadata only (fast)
    meta_df = load_meta(cfg, split="train")

    # Loading arrays + metadata
    feature_arrays, meta_df = load_features(cfg, split="train")

    # Loading a single phoneme array
    from src.data.persist import load_phoneme
    arr = load_phoneme(cfg, split="train", speaker_id="100", sentence_id="1", phoneme_index=3)

Configuration keys
------------------
    features_dir : str
    seed         : int  (logged as attribute for reproducibility)
"""

import io
import logging
from pathlib import Path
from typing import Any

import h5py
import numpy as np
import pandas as pd
from omegaconf import DictConfig

log = logging.getLogger(__name__)

_N_LLD = 25  # eGeMAPS v02 LLD dimensionality (25 features per frame)


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------

def _hdf5_path(features_dir: Path, split: str) -> Path:
    return features_dir / f"{split}_features.h5"


def _serialize_meta(meta_df: pd.DataFrame) -> bytes:
    """Pickle a DataFrame to bytes for embedding in HDF5."""
    buf = io.BytesIO()
    meta_df.to_pickle(buf)
    return buf.getvalue()


def _deserialize_meta(raw: bytes) -> pd.DataFrame:
    """Restore a DataFrame from bytes stored in HDF5."""
    return pd.read_pickle(io.BytesIO(bytes(raw)))


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def save_features(
    feature_arrays: dict[tuple[str, str, int], np.ndarray],
    meta_df: pd.DataFrame,
    cfg: DictConfig,
    split: str,
) -> Path:
    """
    Persist variable-length LLD frame sequences to HDF5.

    Parameters
    ----------
    feature_arrays : dict[(speaker_id, sentence_id, phoneme_index) -> np.ndarray]
        Output of ``extract.extract_features()``.
        Each value is shape ``(T_frames, 23)``.
    meta_df : pd.DataFrame
        Companion metadata DataFrame (output of ``extract.extract_features()``).
        Columns: speaker_id, sentence_id, phoneme_index, phoneme,
                 t_start, t_end, n_frames.
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

    if not feature_arrays:
        raise ValueError("feature_arrays is empty — nothing to save.")

    with h5py.File(out_path, "w") as h5:
        # Top-level metadata attributes
        h5.attrs["split"] = split
        h5.attrs["n_features"] = _N_LLD
        h5.attrs["seed"] = int(cfg.seed)

        for (spk, sent, ph_idx), frames in feature_arrays.items():
            # frames: shape (T_frames, 23)
            if frames.ndim != 2 or frames.shape[1] != _N_LLD:
                log.warning(
                    "Unexpected frame shape %s for (%s, %s, %d). Skipping.",
                    frames.shape, spk, sent, ph_idx,
                )
                continue

            ds_path = f"{spk}/{sent}/{ph_idx}"
            ds = h5.require_dataset(
                ds_path,
                shape=frames.shape,
                dtype=np.float32,
                data=frames.astype(np.float32),
            )
            ds.attrs["n_frames"] = frames.shape[0]

        # Store metadata DataFrame as a pickled blob
        meta_bytes = _serialize_meta(meta_df)
        h5.create_dataset(
            "meta",
            data=np.frombuffer(meta_bytes, dtype=np.uint8),
        )

    n_phonemes = len(feature_arrays)
    total_frames = sum(v.shape[0] for v in feature_arrays.values())
    log.info(
        "Saved %d phoneme LLD sequences (%d total frames) → %s  (split='%s')",
        n_phonemes,
        total_frames,
        out_path,
        split,
    )
    return out_path


def load_meta(cfg: DictConfig, split: str) -> pd.DataFrame:
    """
    Load only the lightweight metadata table for a split (no arrays).

    Parameters
    ----------
    cfg : DictConfig
        Must contain: ``features_dir``.
    split : str
        One of ``"train"``, ``"val"``, ``"test"``.

    Returns
    -------
    pd.DataFrame
        Columns: speaker_id, sentence_id, phoneme_index, phoneme,
                 t_start, t_end, n_frames.
    """
    features_dir = Path(cfg.features_dir)
    h5_path = _hdf5_path(features_dir, split)

    if not h5_path.exists():
        raise FileNotFoundError(
            f"HDF5 feature file not found: {h5_path}. "
            "Run the extraction pipeline first."
        )

    with h5py.File(h5_path, "r") as h5:
        raw = h5["meta"][()]
    return _deserialize_meta(raw.tobytes())


def load_phoneme(
    cfg: DictConfig,
    split: str,
    speaker_id: str,
    sentence_id: str,
    phoneme_index: int,
) -> np.ndarray:
    """
    Load a single phoneme's LLD frame sequence from HDF5.

    Returns
    -------
    np.ndarray of shape ``(T_frames, 23)``
    """
    features_dir = Path(cfg.features_dir)
    h5_path = _hdf5_path(features_dir, split)

    if not h5_path.exists():
        raise FileNotFoundError(f"HDF5 feature file not found: {h5_path}.")

    with h5py.File(h5_path, "r") as h5:
        ds_path = f"{speaker_id}/{sentence_id}/{phoneme_index}"
        if ds_path not in h5:
            raise KeyError(
                f"Phoneme key '{ds_path}' not found in {h5_path}."
            )
        return h5[ds_path][()].astype(np.float32)


def load_features(
    cfg: DictConfig,
    split: str,
) -> tuple[dict[tuple[str, str, int], np.ndarray], pd.DataFrame]:
    """
    Load all LLD frame sequences and metadata for a split.

    Parameters
    ----------
    cfg : DictConfig
        Must contain: ``features_dir``.
    split : str
        One of ``"train"``, ``"val"``, ``"test"``.

    Returns
    -------
    feature_arrays : dict[(speaker_id, sentence_id, phoneme_index) -> np.ndarray]
        Each value is shape ``(T_frames, 23)``.
    meta_df : pd.DataFrame
        Columns: speaker_id, sentence_id, phoneme_index, phoneme,
                 t_start, t_end, n_frames.
    """
    features_dir = Path(cfg.features_dir)
    h5_path = _hdf5_path(features_dir, split)

    if not h5_path.exists():
        raise FileNotFoundError(
            f"HDF5 feature file not found: {h5_path}. "
            "Run the extraction pipeline first."
        )

    feature_arrays: dict[tuple[str, str, int], np.ndarray] = {}

    with h5py.File(h5_path, "r") as h5:
        for spk in h5:
            if spk == "meta":
                continue
            for sent in h5[spk]:
                for ph_idx_str in h5[spk][sent]:
                    arr_key = (spk, sent, int(ph_idx_str))
                    feature_arrays[arr_key] = h5[spk][sent][ph_idx_str][()].astype(
                        np.float32
                    )

        raw = h5["meta"][()]

    meta_df = _deserialize_meta(raw.tobytes())

    if not feature_arrays:
        raise RuntimeError(f"No records loaded from {h5_path}.")

    log.info(
        "Loaded %d phoneme LLD sequences from %s (split='%s')",
        len(feature_arrays),
        h5_path,
        split,
    )
    return feature_arrays, meta_df


def list_splits(cfg: DictConfig) -> list[str]:
    """Return which splits have been persisted to HDF5."""
    features_dir = Path(cfg.features_dir)
    return [
        split
        for split in ("train", "val", "test")
        if _hdf5_path(features_dir, split).exists()
    ]
