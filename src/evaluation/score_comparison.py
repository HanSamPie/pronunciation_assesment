"""
src/evaluation/score_comparison.py
===================================
Cross-metric prediction-accuracy comparison for the ``all_metrics`` score mode.

This module answers the question:
    "How accurately does the model predict *each* scoring axis?"

It accepts the raw ground-truth and prediction arrays produced during
inference, computes PCC / RMSE / SRC for every active metric, then
organises the results into a ``pandas.DataFrame`` that is easy to inspect
and to hand off to the Phase-5 visualisation pipeline.

Typical usage
-------------
::

    from src.evaluation.score_comparison import compare_score_predictions

    report_df = compare_score_predictions(
        targets=targets_dict,       # {metric_name: np.ndarray}
        predictions=predictions_dict,
        score_mode="all_metrics",   # or "major_scores"
        cfg=hydra_cfg,              # optional – used for metric max-score annotation
    )
    print(report_df.to_string(index=False))

The returned DataFrame has the following columns:

=====================  ============================================================
Column                 Description
=====================  ============================================================
``metric``             Canonical metric name (e.g. ``sentence_accuracy``)
``level``              Hierarchy level: ``phoneme``, ``word``, or ``sentence``
``n_samples``          Number of (target, prediction) pairs evaluated
``rmse``               Root-mean-squared error
``pcc``                Pearson correlation coefficient
``max_score``          Upper bound of the metric's score range (from config)
``normalised_rmse``    ``rmse / max_score`` — scale-free error measure
``rank_by_pcc``        Rank of each metric by descending PCC (1 = easiest to predict)
=====================  ============================================================
"""

from __future__ import annotations

import logging
from typing import Dict, Optional

import numpy as np
import pandas as pd
from omegaconf import DictConfig, OmegaConf

from src.evaluation.evaluate import compute_metrics

log = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Level classifier
# ---------------------------------------------------------------------------

_LEVEL_PREFIXES: dict[str, str] = {
    "phoneme_": "phoneme",
    "word_": "word",
    "sentence_": "sentence",
}


def _infer_level(metric_name: str) -> str:
    """Return ``'phoneme'``, ``'word'``, or ``'sentence'`` from the metric name."""
    for prefix, level in _LEVEL_PREFIXES.items():
        if metric_name.startswith(prefix):
            return level
    return "unknown"


# ---------------------------------------------------------------------------
# Max-score lookup
# ---------------------------------------------------------------------------

def _build_max_score_map(cfg: Optional[DictConfig], score_mode: str) -> dict[str, float]:
    """
    Extract the ``max_score`` for every metric in the requested score mode.

    Falls back to ``NaN`` for metrics not found in the config.
    """
    if cfg is None:
        return {}
    try:
        metrics_cfg = cfg.metrics.get(score_mode)
        if metrics_cfg is None:
            log.warning("score_mode '%s' not found in cfg.metrics.", score_mode)
            return {}
        return {
            k: float(v)
            for k, v in OmegaConf.to_container(metrics_cfg, resolve=True).items()
        }
    except Exception as exc:  # pragma: no cover
        log.warning("Could not read max scores from config: %s", exc)
        return {}


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def compare_score_predictions(
    targets: Dict[str, np.ndarray],
    predictions: Dict[str, np.ndarray],
    score_mode: str = "all_metrics",
    cfg: Optional[DictConfig] = None,
) -> pd.DataFrame:
    """
    Compare per-metric prediction accuracy across all active scoring axes.

    Parameters
    ----------
    targets : dict[str, np.ndarray]
        Ground-truth score arrays, keyed by canonical metric name.
        Arrays can be 1-D (flat) or will be flattened automatically.
    predictions : dict[str, np.ndarray]
        Predicted score arrays with the same keys and shapes as *targets*.
    score_mode : str
        Active score mode used for training (``"all_metrics"`` or
        ``"major_scores"``).  Used only for informational annotation in
        the returned DataFrame.
    cfg : DictConfig, optional
        Hydra config object.  When provided, the ``metrics.<score_mode>``
        section is used to annotate ``max_score`` and compute
        ``normalised_rmse``.

    Returns
    -------
    pd.DataFrame
        One row per metric, sorted by descending PCC (most predictable
        first).  See module docstring for column descriptions.

    Raises
    ------
    ValueError
        If *targets* and *predictions* share no common metric keys.
    """
    common_metrics = sorted(set(targets.keys()) & set(predictions.keys()))
    if not common_metrics:
        raise ValueError(
            "targets and predictions share no common metric keys. "
            f"targets keys: {list(targets.keys())}; "
            f"predictions keys: {list(predictions.keys())}"
        )

    missing_in_preds = set(targets.keys()) - set(predictions.keys())
    if missing_in_preds:
        log.warning(
            "The following metrics are in targets but not in predictions "
            "and will be skipped: %s",
            sorted(missing_in_preds),
        )

    max_score_map = _build_max_score_map(cfg, score_mode)

    rows: list[dict] = []
    for metric in common_metrics:
        y_true = np.asarray(targets[metric], dtype=float).ravel()
        y_pred = np.asarray(predictions[metric], dtype=float).ravel()

        # Drop NaN pairs (can arise from padding or missing annotations)
        valid_mask = np.isfinite(y_true) & np.isfinite(y_pred)
        n_dropped = int(np.sum(~valid_mask))
        if n_dropped:
            log.debug(
                "Metric '%s': dropped %d NaN/Inf sample(s) before computing metrics.",
                metric, n_dropped,
            )
        y_true = y_true[valid_mask]
        y_pred = y_pred[valid_mask]

        n_samples = len(y_true)
        if n_samples == 0:
            log.warning("Metric '%s' has no valid samples — skipping.", metric)
            continue

        metric_scores = compute_metrics(y_true, y_pred)
        max_score = max_score_map.get(metric, float("nan"))
        normalised_rmse = (
            metric_scores["rmse"] / max_score
            if np.isfinite(max_score) and max_score > 0
            else float("nan")
        )

        rows.append(
            {
                "metric": metric,
                "level": _infer_level(metric),
                "n_samples": n_samples,
                "rmse": metric_scores["rmse"],
                "pcc": metric_scores["pcc"],
    
                "max_score": max_score,
                "normalised_rmse": normalised_rmse,
            }
        )

    if not rows:
        log.error("No valid metrics could be evaluated.")
        return pd.DataFrame(
            columns=[
                "metric", "level", "n_samples",
                "rmse", "pcc",
                "max_score", "normalised_rmse", "rank_by_pcc",
            ]
        )

    df = pd.DataFrame(rows)

    # Rank metrics by descending PCC (1 = easiest to predict).
    # NaN PCC values are ranked last.
    df["rank_by_pcc"] = (
        df["pcc"]
        .rank(ascending=False, method="min", na_option="bottom")
        .astype(int)
    )

    # Sort by rank (most predictable first)
    df = df.sort_values("rank_by_pcc").reset_index(drop=True)

    log.info(
        "Score comparison complete for score_mode='%s' — %d metric(s) evaluated.",
        score_mode, len(df),
    )
    return df


# ---------------------------------------------------------------------------
# Summary helpers
# ---------------------------------------------------------------------------

def summarise_by_level(df: pd.DataFrame) -> pd.DataFrame:
    """
    Aggregate the per-metric comparison report by hierarchy level.

    Parameters
    ----------
    df : pd.DataFrame
        Output of :func:`compare_score_predictions`.

    Returns
    -------
    pd.DataFrame
        Columns: ``level``, ``n_metrics``, ``mean_pcc``,
        ``mean_rmse``, ``mean_normalised_rmse``.
        Ordered phoneme → word → sentence.
    """
    level_order = ["phoneme", "word", "sentence", "unknown"]
    grouped = (
        df.groupby("level", sort=False)
        .agg(
            n_metrics=("metric", "count"),
            mean_pcc=("pcc", "mean"),
            mean_rmse=("rmse", "mean"),
            mean_normalised_rmse=("normalised_rmse", "mean"),
        )
        .reset_index()
    )
    # Apply canonical level order
    grouped["_order"] = grouped["level"].apply(
        lambda lvl: level_order.index(lvl) if lvl in level_order else len(level_order)
    )
    grouped = grouped.sort_values("_order").drop(columns="_order").reset_index(drop=True)
    return grouped


def format_report(df: pd.DataFrame, float_fmt: str = ".4f") -> str:
    """
    Return a human-readable string representation of the comparison report.

    Parameters
    ----------
    df : pd.DataFrame
        Output of :func:`compare_score_predictions`.
    float_fmt : str
        Format string applied to floating-point columns.

    Returns
    -------
    str
    """
    display_cols = [
        "rank_by_pcc", "metric", "level", "n_samples",
        "pcc", "rmse", "normalised_rmse",
    ]
    available = [c for c in display_cols if c in df.columns]
    return df[available].to_string(index=False, float_format=f"{{:{float_fmt}}}".format)


def main():
    compare_score_predictions()
    summarise_by_level()
    format_report()

if __name__ == "__main__":
    main()
