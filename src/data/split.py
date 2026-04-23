"""
src/data/split.py
=================
Speaker-independent train / validation / test split for Speechocean762.

Strategy
--------
- Group all utterances by speaker_id.
- Shuffle speakers with a fixed global seed.
- Assign speakers to Train (70 %), Val (15 %), Test (15 %) partitions so that
  no single speaker spans more than one partition.
- Write three manifest files (CSV) to ``splits_dir``:
  train_manifest.csv, val_manifest.csv, test_manifest.csv

Each manifest row contains at minimum:
  speaker_id, sentence_id, wav_path, transcript

Usage
-----
    from src.data.split import create_speaker_splits
    create_speaker_splits(cfg)

where ``cfg`` is an OmegaConf / dict-like config with keys:
    data_dir, splits_dir, seed
"""

import random
import logging
import math
from pathlib import Path

import numpy as np
import pandas as pd
from omegaconf import DictConfig

log = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------

def _set_seeds(seed: int) -> None:
    """Set Python, NumPy (and any other global) random seeds."""
    random.seed(seed)
    np.random.seed(seed)


def _parse_kaldi_keyval(path: Path) -> dict[str, str]:
    """Parse a Kaldi-style two-column file (key<TAB|SPACE>value) into a dict."""
    mapping: dict[str, str] = {}
    if not path.exists():
        return mapping
    with open(path, "r") as fh:
        for line in fh:
            line = line.strip()
            if not line:
                continue
            parts = line.split(None, 1)  # split on first whitespace
            if len(parts) == 2:
                mapping[parts[0]] = parts[1]
    return mapping


def _parse_kaldi_text(path: Path) -> dict[str, str]:
    """Parse a Kaldi ``text`` file (utt_id<TAB>transcript) into a dict."""
    return _parse_kaldi_keyval(path)


def _load_scores(data_dir: Path) -> dict:
    """Load ``resource/scores.json`` if available."""
    import json

    scores_path = data_dir / "resource" / "scores.json"
    if scores_path.exists():
        with open(scores_path, "r") as fh:
            return json.load(fh)
    return {}


def _load_metadata(data_dir: Path) -> pd.DataFrame:
    """
    Load the Speechocean762 metadata into a DataFrame.

    The function handles the **native** Speechocean762 directory layout which
    follows Kaldi conventions, as well as a flat ``metadata.csv`` fallback.

    Native Speechocean762 layout (``data_dir``)::

        data/raw/
          WAVE/
            SPEAKER0001/
              000010011.WAV
              ...
          train/
            utt2spk      # utterance_id  speaker_id
            text          # utterance_id  TRANSCRIPT WORDS
            wav.scp       # utterance_id  WAVE/SPEAKERxxxx/utterance_id.WAV
            spk2age       # speaker_id  age
            spk2gender    # speaker_id  m|f
          test/
            (same files as train/)
          resource/
            scores.json   # pronunciation scores keyed by utterance_id

    A ``metadata.csv`` in ``data_dir`` (if present) is used directly.
    Otherwise the function parses the Kaldi-style files to construct it.

    Returns
    -------
    pd.DataFrame with columns:
        speaker_id, sentence_id, wav_path, transcript,
        age (int, if available), gender (str, if available),
        original_split (str – the dataset's own train/test label)
    """
    meta_csv = data_dir / "metadata.csv"
    if meta_csv.exists():
        df = pd.read_csv(meta_csv, dtype={"speaker_id": str, "sentence_id": str})
        log.info("Loaded metadata from %s (%d rows)", meta_csv, len(df))
        return df

    wave_dir = data_dir / "WAVE"

    # ------------------------------------------------------------------
    # Strategy 1 (preferred): parse Kaldi-style files from train/ & test/
    # ------------------------------------------------------------------
    kaldi_dirs = [d for d in [data_dir / "train", data_dir / "test"] if d.is_dir()]

    if kaldi_dirs:
        rows: list[dict] = []

        # Gather demographic info across all sub-directories
        spk2age: dict[str, str] = {}
        spk2gender: dict[str, str] = {}
        for kdir in kaldi_dirs:
            spk2age.update(_parse_kaldi_keyval(kdir / "spk2age"))
            spk2gender.update(_parse_kaldi_keyval(kdir / "spk2gender"))

        # Load pronunciation scores (shared across train/test)
        scores = _load_scores(data_dir)

        for kdir in kaldi_dirs:
            original_split = kdir.name  # "train" or "test"

            utt2spk = _parse_kaldi_keyval(kdir / "utt2spk")
            utt2text = _parse_kaldi_text(kdir / "text")
            wav_scp = _parse_kaldi_keyval(kdir / "wav.scp")

            for utt_id, spk_id in utt2spk.items():
                # Resolve WAV path ─ wav.scp stores paths relative to data_dir
                rel_wav = wav_scp.get(utt_id, "")
                wav_path = (data_dir / rel_wav).resolve() if rel_wav else ""

                # If wav.scp is missing or the file doesn't exist, try to find
                # it by walking the WAVE directory.
                if not rel_wav or not Path(wav_path).exists():
                    if wave_dir.is_dir():
                        # Speaker dirs are named SPEAKERxxxx; spk_id is xxxx
                        candidate = wave_dir / f"SPEAKER{spk_id}" / f"{utt_id}.WAV"
                        if candidate.exists():
                            wav_path = str(candidate.resolve())

                # Extract sentence-level scores if available
                utt_scores: dict[str, object] = {}
                if utt_id in scores:
                    s = scores[utt_id]
                    utt_scores = {
                        "accuracy": s.get("accuracy"),
                        "completeness": s.get("completeness"),
                        "fluency": s.get("fluency"),
                        "prosodic": s.get("prosodic"),
                    }

                row = {
                    "speaker_id": spk_id,
                    "sentence_id": utt_id,
                    "wav_path": str(wav_path),
                    "transcript": utt2text.get(utt_id, ""),
                    "age": int(spk2age[spk_id]) if spk_id in spk2age else None,
                    "gender": spk2gender.get(spk_id, ""),
                    "original_split": original_split,
                    **utt_scores,
                }
                rows.append(row)

        if rows:
            df = pd.DataFrame(rows)
            log.info(
                "Built metadata from Kaldi files: %d utterances, %d speakers, "
                "partitions=%s",
                len(df),
                df["speaker_id"].nunique(),
                sorted(df["original_split"].unique().tolist()),
            )
            return df

    # ------------------------------------------------------------------
    # Strategy 2 (fallback): walk WAVE/ directory tree directly
    # ------------------------------------------------------------------
    if not wave_dir.exists():
        raise FileNotFoundError(
            f"Cannot locate metadata. Expected either {meta_csv}, "
            f"Kaldi-style train/test sub-directories, or a WAVE/ "
            f"sub-directory under {data_dir}."
        )

    rows = []
    for spk_dir in sorted(wave_dir.iterdir()):
        if not spk_dir.is_dir():
            continue
        speaker_id = spk_dir.name
        # Case-insensitive glob: match both .wav and .WAV
        wav_files = sorted(
            list(spk_dir.glob("*.wav")) + list(spk_dir.glob("*.WAV"))
        )
        # Deduplicate (in case filesystem is case-insensitive)
        seen: set[str] = set()
        for wav_file in wav_files:
            key = wav_file.name.lower()
            if key in seen:
                continue
            seen.add(key)
            sentence_id = wav_file.stem
            rows.append(
                {
                    "speaker_id": speaker_id,
                    "sentence_id": sentence_id,
                    "wav_path": str(wav_file.resolve()),
                    "transcript": "",
                }
            )

    if not rows:
        raise RuntimeError(
            f"No WAV files found under {wave_dir}. "
            "Ensure the Speechocean762 dataset is placed at data/raw/."
        )

    df = pd.DataFrame(rows)
    log.info(
        "Built metadata from WAVE directory: %d utterances, %d speakers",
        len(df),
        df["speaker_id"].nunique(),
    )
    return df


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def create_speaker_splits(
    cfg: DictConfig,
    train_ratio: float = 0.70,
    val_ratio: float = 0.15,
    # test_ratio is implicit: 1 - train - val
) -> dict[str, pd.DataFrame]:
    """
    Create speaker-independent Train / Val / Test splits and persist manifests.

    Parameters
    ----------
    cfg : DictConfig
        Must contain: ``data_dir``, ``splits_dir``, ``seed``.
    train_ratio : float
        Fraction of *speakers* allocated to training (default 0.70).
    val_ratio : float
        Fraction of *speakers* allocated to validation (default 0.15).

    Returns
    -------
    dict with keys ``'train'``, ``'val'``, ``'test'``, each a DataFrame.

    Raises
    ------
    ValueError
        If the split ratios do not leave any speakers for the test partition,
        or if a speaker_id appears in more than one partition (sanity check).
    """
    seed: int = int(cfg.seed)
    data_dir = Path(cfg.data_dir)
    splits_dir = Path(cfg.splits_dir)
    splits_dir.mkdir(parents=True, exist_ok=True)

    # ── 1. Seed all global RNGs ──────────────────────────────────────────────
    _set_seeds(seed)

    # ── 2. Load dataset metadata ─────────────────────────────────────────────
    df = _load_metadata(data_dir)

    # ── 3. Shuffle speakers deterministically ────────────────────────────────
    speakers = sorted(df["speaker_id"].unique().tolist())
    rng = random.Random(seed)
    rng.shuffle(speakers)

    n_total = len(speakers)
    n_train = math.floor(train_ratio * n_total)
    n_val = math.floor(val_ratio * n_total)
    n_test = n_total - n_train - n_val

    if n_test <= 0:
        raise ValueError(
            f"With {n_total} speakers and ratios "
            f"train={train_ratio}/val={val_ratio}, the test partition "
            f"would have {n_test} speakers. Adjust ratios or collect more data."
        )

    train_spk = set(speakers[:n_train])
    val_spk = set(speakers[n_train : n_train + n_val])
    test_spk = set(speakers[n_train + n_val :])

    # ── 4. Sanity check — no speaker leakage ─────────────────────────────────
    overlap_tv = train_spk & val_spk
    overlap_tt = train_spk & test_spk
    overlap_vt = val_spk & test_spk
    if overlap_tv or overlap_tt or overlap_vt:
        raise ValueError(
            f"Speaker leakage detected! "
            f"train∩val={overlap_tv}, train∩test={overlap_tt}, "
            f"val∩test={overlap_vt}"
        )

    # ── 5. Build partition DataFrames ─────────────────────────────────────────
    split_col = df["speaker_id"].map(
        {**{s: "train" for s in train_spk},
         **{s: "val" for s in val_spk},
         **{s: "test" for s in test_spk}}
    )
    df = df.copy()
    df["split"] = split_col

    manifests: dict[str, pd.DataFrame] = {}
    for partition in ("train", "val", "test"):
        manifests[partition] = df[df["split"] == partition].copy()

    # ── 6. Persist manifests ──────────────────────────────────────────────────
    for partition, manifest_df in manifests.items():
        out_path = splits_dir / f"{partition}_manifest.csv"
        manifest_df.to_csv(out_path, index=False)
        log.info(
            "Saved %s manifest → %s  (%d utterances, %d speakers)",
            partition,
            out_path,
            len(manifest_df),
            manifest_df["speaker_id"].nunique(),
        )

    # ── 7. Summary log ───────────────────────────────────────────────────────
    log.info(
        "Split summary | train: %d spk / %d utt  |  val: %d spk / %d utt  "
        "|  test: %d spk / %d utt",
        len(train_spk), len(manifests["train"]),
        len(val_spk),   len(manifests["val"]),
        len(test_spk),  len(manifests["test"]),
    )

    return manifests


def load_manifests(splits_dir: Path) -> dict[str, pd.DataFrame]:
    """
    Load previously saved split manifests from ``splits_dir``.

    Returns
    -------
    dict with keys ``'train'``, ``'val'``, ``'test'``.
    """
    manifests = {}
    for partition in ("train", "val", "test"):
        csv_path = splits_dir / f"{partition}_manifest.csv"
        if not csv_path.exists():
            raise FileNotFoundError(
                f"Manifest not found: {csv_path}. "
                "Run create_speaker_splits() first."
            )
        manifests[partition] = pd.read_csv(
            csv_path, dtype={"speaker_id": str, "sentence_id": str}
        )
    return manifests


import hydra
@hydra.main(version_base=None, config_path="../../configs", config_name="base")
def main(cfg: DictConfig) -> None:
    create_speaker_splits(cfg)
if __name__ == "__main__":
    main()