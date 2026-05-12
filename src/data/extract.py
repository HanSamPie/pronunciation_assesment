"""
src/data/extract.py
===================
openSMILE eGeMAPS LLD feature extraction with frame-level phoneme assignment.

Responsibility
--------------
For every WAV file in the manifest, extract the full **eGeMAPS Low-Level
Descriptor (LLD) frame sequence** (25 features/frame) with openSMILE in a
single pass over the waveform.

Each frame carries its own **timestamp** (the centre of the analysis window)
as returned by the openSMILE DataFrame index.  Frames are then assigned to
phonemes by checking whether the **frame midpoint** falls inside a phoneme
interval ``[t_start, t_end)``.

Phoneme boundaries are taken either from ``align.py``
(``parse_textgrids`` / ``run_alignment``) or from any DataFrame that has the
columns ``speaker_id``, ``sentence_id``, ``phoneme``, ``t_start``, ``t_end``.
Passing ``boundary_df=None`` skips the phoneme-assignment step and returns the
raw per-file frame matrices instead (useful for debugging or unsupervised use).

Optional boundary jittering (±5 ms) is applied *to the boundary lookup only*
during training if ``cfg.use_boundary_jitter`` is ``True``; the underlying
audio extraction is always done over the full file.

Usage
-----
    from src.data.extract import extract_features

    # With MFA phoneme boundaries (most common case):
    feature_arrays, meta_df = extract_features(boundary_df, manifest_df, cfg)

    # Without phoneme boundaries (raw per-file frames):
    feature_arrays, meta_df = extract_features(None, manifest_df, cfg)

Return value
------------
When ``boundary_df`` is supplied:
    feature_arrays : dict[(speaker_id, sentence_id, phoneme_index) → np.ndarray]
        Each value has shape ``(T_frames, 25)`` where ``T_frames`` is the
        number of openSMILE frames whose midpoint lies inside the phoneme.
    meta_df : pd.DataFrame
        Columns: speaker_id, sentence_id, phoneme_index, phoneme,
                 t_start, t_end, n_frames.

When ``boundary_df`` is ``None``:
    feature_arrays : dict[(speaker_id, sentence_id) → np.ndarray]
        Full frame matrix per utterance.
    meta_df : pd.DataFrame
        Columns: speaker_id, sentence_id, n_frames.

Configuration keys
------------------
    use_boundary_jitter : bool   enable ±5 ms random boundary shift
    seed                : int    global RNG seed
"""

import logging
import random
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

try:
    import opensmile
except ImportError as exc:
    raise ImportError(
        "opensmile is required. Install with: pip install opensmile"
    ) from exc

try:
    import soundfile as sf
except ImportError as exc:
    raise ImportError(
        "soundfile is required. Install with: pip install soundfile"
    ) from exc

from omegaconf import DictConfig

log = logging.getLogger(__name__)

# Boundary jitter magnitude in seconds
_JITTER_SECONDS = 0.005  # ±5 ms

# eGeMAPS v02 LLD dimensionality (25 features per frame)
_N_LLD = 25


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------

def _set_seeds(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)


def _build_smile() -> opensmile.Smile:
    """Construct an openSMILE instance configured for eGeMAPS LLD extraction."""
    return opensmile.Smile(
        feature_set=opensmile.FeatureSet.eGeMAPSv02,
        feature_level=opensmile.FeatureLevel.LowLevelDescriptors,
        num_workers=1,
        verbose=False,
    )


def _jitter_boundary(
    t_start: float,
    t_end: float,
    duration: float,
    rng: random.Random,
) -> tuple[float, float]:
    """
    Apply random ±5 ms shift to both boundary points.

    The shifted window is clamped to [0, duration] and a minimum width
    of 5 ms is enforced to avoid degenerate extractions.
    """
    shift_start = rng.uniform(-_JITTER_SECONDS, _JITTER_SECONDS)
    shift_end = rng.uniform(-_JITTER_SECONDS, _JITTER_SECONDS)
    new_start = max(0.0, t_start + shift_start)
    new_end = min(duration, t_end + shift_end)
    # Enforce minimum window
    if new_end - new_start < 5e-3:
        new_end = min(duration, new_start + 5e-3)
    return new_start, new_end


def _extract_file_lld(
    smile: opensmile.Smile,
    audio: np.ndarray,
    sr: int,
) -> tuple[np.ndarray, np.ndarray]:
    """
    Extract eGeMAPS LLD frames for an entire waveform in one pass.

    Parameters
    ----------
    audio : np.ndarray
        Mono float32 array of the full waveform.
    sr : int
        Sample rate in Hz.

    Returns
    -------
    frames : np.ndarray, shape (T, 23)
        Feature matrix.  At least one zero row is returned on failure.
    midpoints : np.ndarray, shape (T,)
        Centre time (seconds) of each frame, derived from the openSMILE
        DataFrame index (which stores ``(start, end)`` time tuples).
    """
    try:
        features_df = smile.process_signal(audio, sr)
        # The openSMILE DataFrame has a MultiIndex of (start_time, end_time)
        # expressed as pd.Timedelta.  Compute midpoints in seconds.
        idx = features_df.index
        if isinstance(idx, pd.MultiIndex):
            # Typical openSMILE output: MultiIndex with levels (start, end)
            starts = np.array([t.total_seconds() for t in idx.get_level_values(0)])
            ends = np.array([t.total_seconds() for t in idx.get_level_values(1)])
            midpoints = (starts + ends) / 2.0
        else:
            # Fallback: single-level index assumed to be start times
            midpoints = np.array([t.total_seconds() for t in idx])

        values = features_df.values.astype(np.float32)  # shape (T, n_cols)

        if values.shape[1] != _N_LLD:
            log.warning(
                "Expected %d LLD features, got %d. Padding/truncating columns.",
                _N_LLD,
                values.shape[1],
            )
            if values.shape[1] < _N_LLD:
                pad = np.zeros(
                    (values.shape[0], _N_LLD - values.shape[1]), dtype=np.float32
                )
                values = np.hstack([values, pad])
            else:
                values = values[:, :_N_LLD]

        if values.shape[0] == 0:
            log.warning("No LLD frames returned for audio segment. Returning zero frame.")
            return np.zeros((1, _N_LLD), dtype=np.float32), np.array([0.0]), True

        return values, midpoints, False

    except Exception as exc:  # noqa: BLE001
        log.warning("openSMILE LLD extraction error: %s. Returning zero frame.", exc)
        return np.zeros((1, _N_LLD), dtype=np.float32), np.array([0.0]), True


def _assign_frames_to_phonemes(
    frames: np.ndarray,
    midpoints: np.ndarray,
    phoneme_rows: list[dict],
    duration: float,
    apply_jitter: bool,
    rng: random.Random,
) -> list[tuple[int, np.ndarray, float, float]]:
    """
    Assign openSMILE frames to phoneme intervals by midpoint containment.

    A frame is assigned to phoneme *i* if its midpoint satisfies::

        t_start_i  ≤  midpoint  <  t_end_i

    (For the final phoneme the right boundary is inclusive.)

    Parameters
    ----------
    frames : np.ndarray, shape (T, 23)
    midpoints : np.ndarray, shape (T,)
    phoneme_rows : list of dicts with keys phoneme, t_start, t_end.
    duration : float
        Total waveform duration in seconds (used to clamp jitter).
    apply_jitter : bool
    rng : random.Random

    Returns
    -------
    list of (phoneme_index, frame_block, effective_t_start, effective_t_end)
        ``frame_block`` may be a zero frame if no midpoints fell inside.
    """
    results = []
    n_phonemes = len(phoneme_rows)

    for ph_idx, prow in enumerate(phoneme_rows):
        t_start: float = float(prow["t_start"])
        t_end: float = float(prow["t_end"])

        if apply_jitter:
            t_start, t_end = _jitter_boundary(t_start, t_end, duration, rng)

        # Containment: midpoint in [t_start, t_end)
        # For the last phoneme use t_end inclusive to avoid boundary gaps.
        if ph_idx == n_phonemes - 1:
            mask = (midpoints >= t_start) & (midpoints <= t_end)
        else:
            mask = (midpoints >= t_start) & (midpoints < t_end)

        if mask.any():
            block = frames[mask]
        else:
            log.debug(
                "No frames in phoneme interval [%.4f, %.4f]. Using zero frame.", t_start, t_end
            )
            block = np.zeros((1, _N_LLD), dtype=np.float32)

        results.append((ph_idx, block, t_start, t_end))

    return results


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def extract_features(
    boundary_df: pd.DataFrame | None,
    manifest_df: pd.DataFrame,
    cfg: DictConfig,
    use_jitter: bool | None = None,
) -> tuple[dict, pd.DataFrame]:
    """
    Extract eGeMAPS LLD frame sequences, per file, then assign to phonemes.

    The extraction is done **once per audio file** using openSMILE's own
    internal frame grid.  Frame midpoints are then compared against phoneme
    boundaries to partition the feature matrix.

    Parameters
    ----------
    boundary_df : pd.DataFrame or None
        Phoneme boundary table, e.g. from ``align.parse_textgrids()``.
        Required columns: speaker_id, sentence_id, phoneme, t_start, t_end.
        Pass ``None`` to skip phoneme assignment and return per-file arrays.
    manifest_df : pd.DataFrame
        Split manifest. Required columns: speaker_id, sentence_id, wav_path.
    cfg : DictConfig
        Must contain: ``use_boundary_jitter``, ``seed``.
    use_jitter : bool, optional
        Override ``cfg.use_boundary_jitter``.

    Returns
    -------
    feature_arrays : dict
        * With boundaries: ``{(speaker_id, sentence_id, phoneme_index): np.ndarray}``
          Each value is shape ``(T_frames, 25)`` — variable length per phoneme.
        * Without boundaries: ``{(speaker_id, sentence_id): np.ndarray}``
          Each value is the full file frame matrix.
    meta_df : pd.DataFrame
        * With boundaries: columns speaker_id, sentence_id, phoneme_index,
          phoneme, t_start, t_end, n_frames.
        * Without boundaries: columns speaker_id, sentence_id, n_frames.
    """
    seed: int = int(cfg.seed)
    _set_seeds(seed)

    apply_jitter: bool = (
        use_jitter if use_jitter is not None else bool(cfg.use_boundary_jitter)
    )

    if apply_jitter:
        log.info(
            "Boundary jitter ENABLED (±%d ms, seed=%d)", int(_JITTER_SECONDS * 1000), seed
        )
    else:
        log.info("Boundary jitter DISABLED.")

    # Build wav lookup: (speaker_id, sentence_id) → wav_path
    wav_map: dict[tuple[str, str], str] = {
        (str(row["speaker_id"]), str(row["sentence_id"])): str(row["wav_path"])
        for _, row in manifest_df.iterrows()
    }

    # Build boundary lookup: (speaker_id, sentence_id) → list of phoneme dicts
    # (ordered by t_start within each utterance)
    boundary_map: dict[tuple[str, str], list[dict]] | None = None
    if boundary_df is not None:
        boundary_map = {}
        for (spk, sent), grp in boundary_df.groupby(
            ["speaker_id", "sentence_id"], sort=False
        ):
            boundary_map[(str(spk), str(sent))] = grp.to_dict(orient="records")

    smile = _build_smile()
    rng = random.Random(seed)

    feature_arrays: dict = {}
    meta_records: list[dict[str, Any]] = []

    # Zero-frame accounting
    n_zero_file: int = 0          # files where openSMILE returned nothing
    n_zero_phoneme: int = 0       # phoneme slots with no frames in their interval
    n_total_phoneme: int = 0      # total phoneme slots processed

    for _, row in manifest_df.iterrows():
        spk = str(row["speaker_id"])
        sent = str(row["sentence_id"])
        wav_path = wav_map.get((spk, sent))

        if wav_path is None:
            log.warning("WAV not found for speaker=%s sentence=%s. Skipping.", spk, sent)
            continue

        # ── Load audio ──────────────────────────────────────────────────────
        try:
            audio, sr = sf.read(wav_path, dtype="float32", always_2d=False)
            if audio.ndim > 1:
                audio = audio.mean(axis=1)  # downmix to mono
        except Exception as exc:  # noqa: BLE001
            log.error("Cannot read %s: %s. Skipping utterance.", wav_path, exc)
            continue

        duration = len(audio) / sr

        # ── Extract full-file LLD frames ─────────────────────────────────────
        frames, midpoints, file_was_zero = _extract_file_lld(smile, audio, sr)
        # frames: (T, 23), midpoints: (T,)  — both use openSMILE's own grid
        if file_was_zero:
            n_zero_file += 1

        # ── Assign frames to phonemes (or store as-is) ───────────────────────
        if boundary_map is not None:
            phoneme_rows = boundary_map.get((spk, sent))
            if phoneme_rows is None:
                log.warning(
                    "No phoneme boundaries for speaker=%s sentence=%s. Skipping.", spk, sent
                )
                continue

            assignments = _assign_frames_to_phonemes(
                frames, midpoints, phoneme_rows, duration, apply_jitter, rng
            )

            for ph_idx, block, t_start, t_end in assignments:
                arr_key = (spk, sent, ph_idx)
                feature_arrays[arr_key] = block
                n_total_phoneme += 1
                # A zero block has exactly 1 row of all zeros (fallback)
                is_zero_block = block.shape[0] == 1 and not block.any()
                if is_zero_block:
                    n_zero_phoneme += 1
                meta_records.append(
                    {
                        "speaker_id": spk,
                        "sentence_id": sent,
                        "phoneme_index": ph_idx,
                        "phoneme": str(phoneme_rows[ph_idx]["phoneme"]),
                        "t_start": t_start,
                        "t_end": t_end,
                        "n_frames": block.shape[0],
                        "zero_frame": is_zero_block,
                    }
                )

        else:
            # No phoneme boundaries — store full file matrix
            arr_key = (spk, sent)
            feature_arrays[arr_key] = frames
            meta_records.append(
                {
                    "speaker_id": spk,
                    "sentence_id": sent,
                    "n_frames": frames.shape[0],
                }
            )

    if not feature_arrays:
        raise RuntimeError(
            "No feature records were extracted. Check boundary_df and manifest_df."
        )

    meta_df = pd.DataFrame(meta_records)

    if boundary_map is not None:
        zero_pct = (
            100.0 * n_zero_phoneme / n_total_phoneme if n_total_phoneme > 0 else 0.0
        )
        log.info(
            "Extracted %d phoneme LLD sequences (%d utterances, %d speakers). "
            "Avg frames/phoneme: %.1f | "
            "Zero-frame phonemes: %d / %d (%.1f%%) | "
            "Zero-frame files (openSMILE failure): %d",
            len(meta_df),
            meta_df[["speaker_id", "sentence_id"]].drop_duplicates().shape[0],
            meta_df["speaker_id"].nunique(),
            meta_df["n_frames"].mean(),
            n_zero_phoneme,
            n_total_phoneme,
            zero_pct,
            n_zero_file,
        )
        if n_zero_phoneme > 0:
            log.warning(
                "%d phoneme slots (%.1f%%) received no frames — "
                "check phoneme boundary coverage vs. openSMILE frame grid.",
                n_zero_phoneme,
                zero_pct,
            )
    else:
        log.info(
            "Extracted %d file-level LLD matrices (%d speakers). "
            "Zero-frame files (openSMILE failure): %d",
            len(meta_df),
            meta_df["speaker_id"].nunique(),
            n_zero_file,
        )

    return feature_arrays, meta_df


# ---------------------------------------------------------------------------
# CLI entry-point (hydra)
# ---------------------------------------------------------------------------

import hydra


@hydra.main(version_base=None, config_path="../../configs", config_name="base")
def main(cfg: DictConfig) -> None:
    from src.data.split import load_manifests
    from src.data.align_dataset import load_boundaries
    from src.data.persist import save_features

    splits_dir = Path(cfg.splits_dir)
    manifests = load_manifests(splits_dir)

    for partition, manifest_df in manifests.items():
        log.info(
            "Extracting LLD features for partition '%s' "
            "(alignment_source='%s')",
            partition,
            getattr(cfg, 'alignment_source', 'dataset'),
        )

        try:
            boundary_df = load_boundaries(manifest_df, cfg)
        except (FileNotFoundError, RuntimeError) as exc:
            log.warning(
                "Could not load phoneme boundaries: %s. "
                "Falling back to per-file extraction (no phoneme assignment).",
                exc,
            )
            boundary_df = None

        feature_arrays, meta_df = extract_features(
            boundary_df,
            manifest_df,
            cfg,
            use_jitter=(partition == "train" and cfg.use_boundary_jitter),
        )
        save_features(feature_arrays, meta_df, cfg, split=partition)


if __name__ == "__main__":
    main()

