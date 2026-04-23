"""
src/data/align.py
=================
Wrapper around the **Montreal Forced Aligner (MFA)** CLI.

Responsibility
--------------
Given a split manifest (a DataFrame with ``wav_path`` and ``transcript``
columns), this module:
1. Constructs the MFA input corpus directory layout.
2. Calls ``mfa align`` with the ``english_mfa`` acoustic model.
3. Parses the resulting TextGrid files.
4. Returns a DataFrame with one row per *phoneme* containing:
   speaker_id, sentence_id, phoneme, t_start, t_end

Usage
-----
    from src.data.align import run_alignment, parse_textgrids
    boundary_df = run_alignment(manifest_df, cfg)
    # or load previously computed TextGrids:
    boundary_df = parse_textgrids(textgrid_dir, manifest_df)

Configuration keys used from ``cfg``
--------------------------------------
    mfa_model   : str  (default "english_mfa")
    data_dir    : str  root data directory
    seed        : int  (not directly used but documented for completeness)

MFA must be installed and available on PATH.
  conda install -c conda-forge montreal-forced-aligner
  mfa model download acoustic english_mfa
  mfa model download dictionary english_mfa
"""

import logging
import os
import re
import shutil
import subprocess
import tempfile
from pathlib import Path

import numpy as np
import pandas as pd
from omegaconf import DictConfig

log = logging.getLogger(__name__)

_SILENCE_LABELS = frozenset({"sp", "spn", "sil", "<eps>", ""})


# ---------------------------------------------------------------------------
# TextGrid parsing
# ---------------------------------------------------------------------------

def _parse_single_textgrid(tg_path: Path, speaker_id: str, sentence_id: str) -> list[dict]:
    """
    Parse one TextGrid file and extract phone-tier intervals.

    Returns a list of dicts with keys:
        speaker_id, sentence_id, phoneme, t_start, t_end
    """
    records: list[dict] = []

    text = tg_path.read_text(encoding="utf-8", errors="replace")

    # Locate the "phones" or "phone" tier
    # TextGrid format is line-based; we do a simple state-machine parse.
    in_phone_tier = False
    in_intervals = False
    buffer: dict = {}

    for line in text.splitlines():
        line_stripped = line.strip()

        name_match = re.search(r'name\s*=\s*"(.+?)"', line_stripped, re.IGNORECASE)
        if name_match:
            tier_name = name_match.group(1).lower()
            in_phone_tier = tier_name in ("phone", "phones")
            in_intervals = False
            continue

        if in_phone_tier:
            if line_stripped.startswith("intervals ["):
                in_intervals = True
                buffer = {}
                continue

            if in_intervals:
                if line_stripped.startswith("xmin"):
                    val = line_stripped.split("=", 1)[1].strip()
                    buffer["t_start"] = float(val)
                elif line_stripped.startswith("xmax"):
                    val = line_stripped.split("=", 1)[1].strip()
                    buffer["t_end"] = float(val)
                elif line_stripped.startswith("text"):
                    val = line_stripped.split("=", 1)[1].strip().strip('"')
                    buffer["phoneme"] = val
                    # All three fields collected → emit record
                    if buffer.get("phoneme") not in _SILENCE_LABELS and buffer["phoneme"]:
                        records.append(
                            {
                                "speaker_id": speaker_id,
                                "sentence_id": sentence_id,
                                "phoneme": buffer["phoneme"],
                                "t_start": buffer["t_start"],
                                "t_end": buffer["t_end"],
                            }
                        )
                    buffer = {}
                    in_intervals = False

    if not records:
        log.warning(
            "No phoneme intervals found in %s (speaker=%s, sentence=%s)",
            tg_path,
            speaker_id,
            sentence_id,
        )
    return records


def parse_textgrids(
    textgrid_dir: Path,
    manifest_df: pd.DataFrame,
) -> pd.DataFrame:
    """
    Walk ``textgrid_dir`` and parse all TextGrid files that correspond to
    utterances in ``manifest_df``.

    Parameters
    ----------
    textgrid_dir : Path
        Root directory produced by ``mfa align``.
    manifest_df : pd.DataFrame
        Must have columns: speaker_id, sentence_id.

    Returns
    -------
    pd.DataFrame with columns: speaker_id, sentence_id, phoneme, t_start, t_end
    """
    utt_set = set(
        zip(manifest_df["speaker_id"].astype(str), manifest_df["sentence_id"].astype(str))
    )

    all_records: list[dict] = []
    tg_paths = list(textgrid_dir.rglob("*.TextGrid"))

    if not tg_paths:
        raise FileNotFoundError(
            f"No TextGrid files found under {textgrid_dir}. "
            "Run MFA alignment first."
        )

    for tg_path in sorted(tg_paths):
        # Infer speaker_id and sentence_id from directory structure:
        #   <textgrid_dir>/<speaker_id>/<sentence_id>.TextGrid
        sentence_id = tg_path.stem
        speaker_id = tg_path.parent.name

        if (speaker_id, sentence_id) not in utt_set:
            continue  # not in this manifest partition

        records = _parse_single_textgrid(tg_path, speaker_id, sentence_id)
        all_records.extend(records)

    if not all_records:
        raise RuntimeError(
            "Parsed 0 phoneme records from TextGrids under "
            f"{textgrid_dir}. Check MFA output."
        )

    df = pd.DataFrame(all_records)
    log.info(
        "Parsed %d phoneme intervals from %d TextGrid files",
        len(df),
        len(tg_paths),
    )
    return df


# ---------------------------------------------------------------------------
# MFA invocation
# ---------------------------------------------------------------------------

def _build_corpus_dir(manifest_df: pd.DataFrame, corpus_dir: Path) -> None:
    """
    Stage audio + transcript files in the flat corpus layout expected by MFA.

    Layout::
        <corpus_dir>/
          <speaker_id>/
            <sentence_id>.wav    (symlink or copy)
            <sentence_id>.lab    (transcript text)
    """
    corpus_dir.mkdir(parents=True, exist_ok=True)

    for _, row in manifest_df.iterrows():
        spk_dir = corpus_dir / str(row["speaker_id"])
        spk_dir.mkdir(exist_ok=True)

        wav_src = Path(row["wav_path"])
        wav_dst = spk_dir / f"{row['sentence_id']}.wav"
        if not wav_dst.exists():
            try:
                wav_dst.symlink_to(wav_src.resolve())
            except OSError:
                shutil.copy2(wav_src, wav_dst)

        lab_dst = spk_dir / f"{row['sentence_id']}.lab"
        transcript = str(row.get("transcript", "")).strip()
        lab_dst.write_text(transcript, encoding="utf-8")


def run_alignment(
    manifest_df: pd.DataFrame,
    cfg: DictConfig,
    textgrid_out_dir: Path | None = None,
    num_jobs: int | None = None,
) -> pd.DataFrame:
    """
    Run MFA forced alignment for all utterances in ``manifest_df``.

    Parameters
    ----------
    manifest_df : pd.DataFrame
        Columns: speaker_id, sentence_id, wav_path, transcript.
    cfg : DictConfig
        Must contain: ``mfa_model``, ``data_dir``.
    textgrid_out_dir : Path, optional
        Where the MFA TextGrid output will be stored.
        Defaults to ``<data_dir>/textgrids``.
    num_jobs : int, optional
        Parallelism passed to ``mfa align --jobs``.
        Defaults to half the available CPU count.

    Returns
    -------
    pd.DataFrame  — phoneme boundary table (speaker, sentence, phoneme, t_start, t_end).
    """
    mfa_model: str = cfg.mfa_model  # e.g. "english_mfa"
    data_dir = Path(cfg.data_dir)

    if textgrid_out_dir is None:
        textgrid_out_dir = data_dir.parent / "textgrids"
    textgrid_out_dir.mkdir(parents=True, exist_ok=True)

    if num_jobs is None:
        num_jobs = max(1, (os.cpu_count() or 2) // 2)

    with tempfile.TemporaryDirectory(prefix="mfa_corpus_") as tmp:
        corpus_dir = Path(tmp) / "corpus"
        _build_corpus_dir(manifest_df, corpus_dir)

        cmd = [
            "mfa",
            "align",
            str(corpus_dir),
            mfa_model,   # dictionary identifier (english_mfa)
            mfa_model,   # acoustic model identifier (english_mfa)
            str(textgrid_out_dir),
            "--jobs",
            str(num_jobs),
            "--clean",
        ]

        log.info("Running MFA: %s", " ".join(cmd))
        result = subprocess.run(
            cmd,
            capture_output=True,
            text=True,
            check=False,
        )

        if result.returncode != 0:
            log.error("MFA stderr:\n%s", result.stderr)
            raise RuntimeError(
                f"MFA alignment failed (exit code {result.returncode}). "
                f"stderr: {result.stderr[-2000:]}"
            )

        log.info("MFA alignment complete. Output → %s", textgrid_out_dir)

    return parse_textgrids(textgrid_out_dir, manifest_df)


import hydra
@hydra.main(version_base=None, config_path="../../configs", config_name="base")
def main(cfg: DictConfig) -> None:
    from src.data.split import load_manifests
    from pathlib import Path
    import pandas as pd
    
    manifests = load_manifests(Path(cfg.splits_dir))
    manifest_df = pd.concat(manifests.values(), ignore_index=True)
    run_alignment(manifest_df, cfg)
if __name__ == "__main__":
    main()