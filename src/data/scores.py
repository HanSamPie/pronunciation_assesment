"""
src/data/scores.py
==================
Loader for the Speechocean762 hierarchical score annotations.

Speechocean762 ships ``scores.json`` with ground-truth human ratings at three
levels of granularity:

* **Sentence-level** — accuracy, completeness, fluency, prosodic, total
* **Word-level**     — accuracy (0–10), stress (5–10), total
* **Phoneme-level**  — phones-accuracy (0.0–2.0 per phone)

This module loads the JSON once and exposes a fast lookup keyed by sentence ID
so the training dataset can attach per-phoneme and per-word targets to every
sample without repeated I/O.

Usage
-----
    from src.data.scores import load_hierarchical_scores

    scores = load_hierarchical_scores(Path("data/raw/resource/scores.json"))
    entry  = scores["000010011"]

    # Sentence-level
    entry["accuracy"]       # 8

    # Word-level
    entry["words"][0]["accuracy"]   # 10
    entry["words"][0]["stress"]     # 10

    # Phoneme-level
    entry["words"][0]["phones-accuracy"]  # [2.0, 2.0]
"""

import json
import logging
from pathlib import Path

log = logging.getLogger(__name__)


def load_hierarchical_scores(scores_path: Path) -> dict[str, dict]:
    """Load ``scores.json`` and return the raw dict keyed by sentence ID.

    Parameters
    ----------
    scores_path : Path
        Absolute or relative path to the Speechocean762 ``scores.json`` file
        (typically ``data/raw/resource/scores.json``).

    Returns
    -------
    dict[str, dict]
        Mapping from 9-digit zero-padded sentence ID string to the full
        annotation dict containing sentence-, word-, and phoneme-level scores.

    Raises
    ------
    FileNotFoundError
        If *scores_path* does not exist.
    """
    scores_path = Path(scores_path)
    if not scores_path.exists():
        raise FileNotFoundError(
            f"Scores file not found: {scores_path}. "
            "Expected Speechocean762 scores.json."
        )

    with open(scores_path, "r", encoding="utf-8") as fh:
        raw: dict[str, dict] = json.load(fh)

    # Ensure keys are zero-padded to 9 digits for consistent lookup.
    scores = {str(k).zfill(9): v for k, v in raw.items()}

    log.info(
        "Loaded hierarchical scores for %d sentences from %s",
        len(scores),
        scores_path,
    )
    return scores


def flat_phoneme_scores(entry: dict) -> list[float]:
    """Extract a flat list of per-phoneme accuracy scores from a sentence entry.

    The phonemes are returned in word order, matching the phoneme index
    sequence stored in the HDF5 feature files.

    Parameters
    ----------
    entry : dict
        A single sentence entry from ``scores.json``.

    Returns
    -------
    list[float]
        One accuracy value (0.0–2.0) per phoneme.
    """
    result: list[float] = []
    for word in entry.get("words", []):
        result.extend(word.get("phones-accuracy", []))
    return result


def word_boundaries_from_entry(entry: dict) -> list[tuple[int, int]]:
    """Derive word boundary indices from the phoneme-per-word structure.

    Parameters
    ----------
    entry : dict
        A single sentence entry from ``scores.json``.

    Returns
    -------
    list[tuple[int, int]]
        ``(start, end)`` index pairs (exclusive end) for each word,
        indicating which contiguous phoneme indices belong to that word.
    """
    boundaries: list[tuple[int, int]] = []
    offset = 0
    for word in entry.get("words", []):
        n_phones = len(word.get("phones", []))
        boundaries.append((offset, offset + n_phones))
        offset += n_phones
    return boundaries


def word_scores(entry: dict, metric: str = "accuracy") -> list[float]:
    """Extract per-word scores for a given metric.

    Parameters
    ----------
    entry : dict
        A single sentence entry from ``scores.json``.
    metric : str
        Word-level metric key (``"accuracy"`` or ``"stress"``).

    Returns
    -------
    list[float]
        One score per word.
    """
    return [float(w.get(metric, 0.0)) for w in entry.get("words", [])]
