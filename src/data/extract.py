"""
src/data/extract.py
===================
openSMILE eGeMAPS feature extraction bounded by MFA phoneme timestamps.

Responsibility
--------------
For every phoneme boundary row produced by ``align.py``, slice the
corresponding WAV audio within [t_start, t_end] and extract the 88-dimensional
eGeMAPS feature vector using the ``opensmile`` Python package.

Optional boundary jittering (±5 ms) is applied during training if
``cfg.use_boundary_jitter`` is ``True``.

Usage
-----
    from src.data.extract import extract_features
    feature_df = extract_features(boundary_df, manifest_df, cfg)

Returned DataFrame columns
--------------------------
    speaker_id, sentence_id, phoneme_index, phoneme,
    t_start, t_end,
    feat_0 … feat_87   (88 eGeMAPS values)

Configuration keys
------------------
    use_boundary_jitter : bool   enable ±5ms random boundary shift
    seed                : int    global RNG seed
"""

import logging
import random
from pathlib import Path

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

# Number of eGeMAPS features
_N_FEATURES = 88
_FEATURE_COLS = [f"feat_{i}" for i in range(_N_FEATURES)]


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------

def _set_seeds(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)


def _build_smile() -> opensmile.Smile:
    """Construct an openSMILE instance configured for eGeMAPS extraction."""
    return opensmile.Smile(
        feature_set=opensmile.FeatureSet.eGeMAPSv02,
        feature_level=opensmile.FeatureLevel.Functionals,
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


def _extract_segment(
    smile: opensmile.Smile,
    audio: np.ndarray,
    sr: int,
    t_start: float,
    t_end: float,
) -> np.ndarray:
    """
    Extract eGeMAPS features from a sub-segment of ``audio``.

    Parameters
    ----------
    audio : np.ndarray
        Mono float32 array of the full waveform.
    sr : int
        Sample rate in Hz.
    t_start, t_end : float
        Segment boundaries in seconds.

    Returns
    -------
    np.ndarray of shape (88,)
    """
    s_start = max(0, int(t_start * sr))
    s_end = min(len(audio), int(t_end * sr))

    if s_end <= s_start:
        log.warning(
            "Empty segment [%f, %f] → returning zero vector.", t_start, t_end
        )
        return np.zeros(_N_FEATURES, dtype=np.float32)

    segment = audio[s_start:s_end]

    try:
        features_df = smile.process_signal(segment, sr)
        values = features_df.values.flatten()
        if len(values) != _N_FEATURES:
            log.warning(
                "Expected %d features, got %d. Padding/truncating.",
                _N_FEATURES,
                len(values),
            )
            if len(values) < _N_FEATURES:
                values = np.pad(values, (0, _N_FEATURES - len(values)))
            else:
                values = values[:_N_FEATURES]
        return values.astype(np.float32)
    except Exception as exc:  # noqa: BLE001
        log.warning("openSMILE extraction error: %s. Returning zeros.", exc)
        return np.zeros(_N_FEATURES, dtype=np.float32)


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def extract_features(
    boundary_df: pd.DataFrame,
    manifest_df: pd.DataFrame,
    cfg: DictConfig,
    use_jitter: bool | None = None,
) -> pd.DataFrame:
    """
    Extract eGeMAPS features for every phoneme in ``boundary_df``.

    Parameters
    ----------
    boundary_df : pd.DataFrame
        Output of ``align.parse_textgrids()``.
        Required columns: speaker_id, sentence_id, phoneme, t_start, t_end.
    manifest_df : pd.DataFrame
        Split manifest. Required columns: speaker_id, sentence_id, wav_path.
    cfg : DictConfig
        Must contain: ``use_boundary_jitter``, ``seed``.
    use_jitter : bool, optional
        Override ``cfg.use_boundary_jitter``.

    Returns
    -------
    pd.DataFrame with columns:
        speaker_id, sentence_id, phoneme_index, phoneme,
        t_start, t_end, feat_0 … feat_87
    """
    seed: int = int(cfg.seed)
    _set_seeds(seed)

    apply_jitter: bool = use_jitter if use_jitter is not None else bool(cfg.use_boundary_jitter)

    if apply_jitter:
        log.info("Boundary jitter ENABLED (±%d ms, seed=%d)", int(_JITTER_SECONDS * 1000), seed)
    else:
        log.info("Boundary jitter DISABLED.")

    # Build wav lookup: (speaker_id, sentence_id) → wav_path
    wav_map: dict[tuple[str, str], str] = {
        (str(row["speaker_id"]), str(row["sentence_id"])): str(row["wav_path"])
        for _, row in manifest_df.iterrows()
    }

    smile = _build_smile()
    rng = random.Random(seed)

    # Cache loaded audio to avoid re-reading the same WAV for every phoneme
    audio_cache: dict[str, tuple[np.ndarray, int]] = {}

    records: list[dict] = []

    # Group by utterance to track phoneme_index within each sentence
    for (spk, sent), group in boundary_df.groupby(
        ["speaker_id", "sentence_id"], sort=False
    ):
        key = (str(spk), str(sent))
        wav_path = wav_map.get(key)
        if wav_path is None:
            log.warning(
                "WAV not found for speaker=%s sentence=%s. Skipping.", spk, sent
            )
            continue

        # Load audio (cached)
        if wav_path not in audio_cache:
            try:
                audio, sr = sf.read(wav_path, dtype="float32", always_2d=False)
                if audio.ndim > 1:
                    audio = audio.mean(axis=1)  # downmix to mono
                audio_cache[wav_path] = (audio, sr)
            except Exception as exc:  # noqa: BLE001
                log.error("Cannot read %s: %s. Skipping utterance.", wav_path, exc)
                continue

        audio, sr = audio_cache[wav_path]
        duration = len(audio) / sr

        for phoneme_index, (_, prow) in enumerate(group.iterrows()):
            t_start: float = float(prow["t_start"])
            t_end: float = float(prow["t_end"])

            if apply_jitter:
                t_start, t_end = _jitter_boundary(t_start, t_end, duration, rng)

            feats = _extract_segment(smile, audio, sr, t_start, t_end)

            record = {
                "speaker_id": spk,
                "sentence_id": sent,
                "phoneme_index": phoneme_index,
                "phoneme": prow["phoneme"],
                "t_start": t_start,
                "t_end": t_end,
            }
            for fi, fv in enumerate(feats):
                record[f"feat_{fi}"] = fv
            records.append(record)

    if not records:
        raise RuntimeError(
            "No feature records were extracted. Check boundary_df and manifest_df."
        )

    result_df = pd.DataFrame(records)
    log.info(
        "Extracted %d phoneme feature vectors (%d utterances, %d speakers)",
        len(result_df),
        result_df[["speaker_id", "sentence_id"]].drop_duplicates().shape[0],
        result_df["speaker_id"].nunique(),
    )
    return result_df
