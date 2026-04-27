"""
src/data/align_dataset.py
=========================
Phoneme boundary loader for the **pre-computed TextGrid alignments** that are
bundled with the Speechocean762 dataset.

Background
----------
The Speechocean762 dataset ships a set of Praat TextGrid files containing
word- and phone-level time alignments produced by an external forced-aligner.
These live under::

    data/textgrids/
        <speaker_id>/
            <sentence_id>.TextGrid

This module reads those files and returns the same ``pd.DataFrame`` schema
that ``align.py`` produces, so every downstream consumer (``extract.py``,
``persist.py``, analysis scripts) can use either source transparently.

Output schema (identical to ``align.py``)
------------------------------------------
One row per non-silence phoneme:

    speaker_id  : str   four-digit speaker identifier (e.g. "0001")
    sentence_id : str   nine-digit utterance identifier (e.g. "000010011")
    phoneme     : str   phoneme label as written in the TextGrid
    t_start     : float interval start time in seconds
    t_end       : float interval end time in seconds

Usage
-----
    from src.data.align_dataset import load_dataset_boundaries

    boundary_df = load_dataset_boundaries(textgrid_dir, manifest_df)

Configuration key (used by ``extract.py`` CLI)
----------------------------------------------
    alignment_source : str   "dataset" (default) | "mfa"
    textgrid_dir     : str   path to the TextGrid root directory
                             default: <data_dir>/../textgrids
                             (i.e. data/textgrids relative to the project root)
"""

import logging
from pathlib import Path

import pandas as pd
from omegaconf import DictConfig

# Reuse the TextGrid parsing logic already implemented in align.py.
# We do NOT import run_alignment — this module never touches MFA.
from src.data.align import parse_textgrids  # noqa: E402

log = logging.getLogger(__name__)


def load_dataset_boundaries(
    textgrid_dir: Path,
    manifest_df: pd.DataFrame,
) -> pd.DataFrame:
    """
    Load phoneme boundaries from the dataset-bundled TextGrid files.

    This is a thin, named wrapper around ``align.parse_textgrids`` that makes
    the intent explicit: we are reading the *dataset's own* alignments, not
    alignments produced by running MFA ourselves.

    Parameters
    ----------
    textgrid_dir : Path
        Root directory of the pre-computed TextGrids.  The expected layout is::

            <textgrid_dir>/
                <speaker_id>/
                    <sentence_id>.TextGrid

    manifest_df : pd.DataFrame
        Split manifest. Required columns: speaker_id, sentence_id.

    Returns
    -------
    pd.DataFrame
        Columns: speaker_id, sentence_id, phoneme, t_start, t_end.
        Silence intervals are excluded.

    Raises
    ------
    FileNotFoundError
        If ``textgrid_dir`` does not exist or contains no TextGrid files.
    RuntimeError
        If no phoneme intervals could be parsed from the TextGrid files.
    """
    if not textgrid_dir.exists():
        raise FileNotFoundError(
            f"Dataset TextGrid directory not found: {textgrid_dir}. "
            "Ensure the pre-computed TextGrids are present at the expected path "
            "or adjust 'textgrid_dir' in your config."
        )

    log.info(
        "Loading dataset phoneme boundaries from %s  (%d utterances in manifest)",
        textgrid_dir,
        len(manifest_df),
    )

    boundary_df = parse_textgrids(textgrid_dir, manifest_df)

    log.info(
        "Dataset alignment loaded: %d phoneme intervals, %d utterances, %d speakers",
        len(boundary_df),
        boundary_df[["speaker_id", "sentence_id"]].drop_duplicates().shape[0],
        boundary_df["speaker_id"].nunique(),
    )
    return boundary_df


def load_boundaries(
    manifest_df: pd.DataFrame,
    cfg: DictConfig,
) -> pd.DataFrame:
    """
    Unified boundary loader — dispatches to the dataset or MFA source based on
    ``cfg.alignment_source``.

    Parameters
    ----------
    manifest_df : pd.DataFrame
        Split manifest with columns: speaker_id, sentence_id, wav_path, transcript.
    cfg : DictConfig
        Expected keys:

        ``alignment_source`` : str
            ``"dataset"`` (default) — read from pre-computed TextGrids.
            ``"mfa"``              — run or read MFA-produced TextGrids.
        ``textgrid_dir`` : str, optional
            Explicit path to the TextGrid root directory.
            Falls back to ``<data_dir>/../textgrids`` if not set.
        ``data_dir`` : str
            Required when ``textgrid_dir`` is not set.

    Returns
    -------
    pd.DataFrame
        Columns: speaker_id, sentence_id, phoneme, t_start, t_end.
    """
    source: str = str(getattr(cfg, "alignment_source", "dataset")).lower()

    # Resolve textgrid directory
    if hasattr(cfg, "textgrid_dir") and cfg.textgrid_dir:
        textgrid_dir = Path(cfg.textgrid_dir)
    else:
        data_dir = Path(cfg.data_dir)
        textgrid_dir = data_dir.parent / "textgrids"

    if source == "dataset":
        return load_dataset_boundaries(textgrid_dir, manifest_df)

    elif source == "mfa":
        from src.data.align import parse_textgrids as _parse_mfa

        log.info("Using MFA alignment source from %s", textgrid_dir)

        if not textgrid_dir.exists():
            # MFA has not been run yet — run it now
            log.warning(
                "MFA TextGrid directory not found (%s). Running MFA alignment now.",
                textgrid_dir,
            )
            from src.data.align import run_alignment
            return run_alignment(manifest_df, cfg, textgrid_out_dir=textgrid_dir)

        return _parse_mfa(textgrid_dir, manifest_df)

    else:
        raise ValueError(
            f"Unknown alignment_source '{source}'. "
            "Must be 'dataset' or 'mfa'."
        )
