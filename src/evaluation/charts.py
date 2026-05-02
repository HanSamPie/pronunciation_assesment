"""
src/evaluation/charts.py
=========================
Publication-quality evaluation charts with dark theme.

9 chart types, each saved with split-identifiable filenames into
per-split output folders.
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Dict, Optional

import matplotlib
matplotlib.use("Agg")

import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
import numpy as np
import pandas as pd
import seaborn as sns
from scipy.stats import pearsonr

log = logging.getLogger(__name__)

# ── Default theme ──────────────────────────────────────────────────────
PALETTE = {"train": "#3b82f6", "val": "#f59e0b", "test": "#10b981"}
BG_COLOR    = "white"
CARD_BG     = "white"
TEXT_COLOR  = "black"
GRID_COLOR  = "#e5e7eb"
ACCENT      = "#3b82f6"

MODEL_COLORS = {
    "linear": "#3b82f6",
    "tree":   "#10b981",
}

def _bigru_color(idx: int) -> str:
    palette = ["#ec4899", "#f97316", "#8b5cf6", "#eab308", "#0ea5e9", "#22c55e"]
    return palette[idx % len(palette)]

plt.rcParams.update({
    "axes.grid": True, "grid.color": GRID_COLOR, "grid.alpha": 0.4,
    "font.family": "DejaVu Sans", "font.size": 11,
    "savefig.bbox": "tight", "savefig.dpi": 200,
})

PROJECT_ROOT = Path(__file__).resolve().parents[2]


def _save(fig, path: Path):
    path = Path(path).resolve()
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path)
    plt.close(fig)
    try:
        display = path.relative_to(PROJECT_ROOT)
    except ValueError:
        display = path
    print(f"  ✓ {display}")


def _model_color(name: str, idx: int = 0) -> str:
    if name in MODEL_COLORS:
        return MODEL_COLORS[name]
    return _bigru_color(idx)


# ═══════════════════════════════════════════════════════════════════════
# 1. Scatter: Predicted vs Ground Truth
# ═══════════════════════════════════════════════════════════════════════

def plot_scatter_predictions(
    targets: dict, predictions: dict, model_name: str,
    split: str, metrics: list[str], output_dir: Path,
):
    avail = [m for m in metrics if m in targets and m in predictions]
    if not avail:
        return
    n = len(avail)
    cols = min(n, 3)
    rows = (n + cols - 1) // cols
    fig, axes = plt.subplots(rows, cols, figsize=(6 * cols, 5.5 * rows), squeeze=False)
    fig.suptitle(f"Predicted vs Ground Truth — {model_name} [{split}]\n(PCC: higher is better)",
                 fontsize=16, fontweight="bold", y=1.06)

    for i, metric in enumerate(avail):
        ax = axes[i // cols][i % cols]
        yt = np.asarray(targets[metric]).ravel()
        yp = np.asarray(predictions[metric]).ravel()
        ax.scatter(yt, yp, alpha=0.25, s=8, c=ACCENT, edgecolors="none")
        lo = min(yt.min(), yp.min()) - 0.5
        hi = max(yt.max(), yp.max()) + 0.5
        ax.plot([lo, hi], [lo, hi], "--", color="#f97316", linewidth=1.5, label="y=x")
        try:
            pcc, _ = pearsonr(yt, yp)
            ax.set_title(f"{metric}  (PCC={pcc:.3f})", fontsize=11, fontweight="bold")
        except Exception:
            ax.set_title(metric, fontsize=11, fontweight="bold")
        ax.set_xlabel("Ground Truth")
        ax.set_ylabel("Predicted")
        ax.legend(fontsize=8)

    # hide unused axes
    for j in range(n, rows * cols):
        axes[j // cols][j % cols].set_visible(False)

    fig.tight_layout()
    model_type = model_name.split('_')[0]
    _save(fig, output_dir / split / model_type / f"scatter_predictions_{model_name}_{split}.png")


# ═══════════════════════════════════════════════════════════════════════
# 2. Metrics bar chart (single model)
# ═══════════════════════════════════════════════════════════════════════

def plot_metrics_bar(
    results: dict, model_name: str, split: str, output_dir: Path,
):
    if not results:
        return
    metrics = list(results.keys())
    pcc_vals = [results[m].get("pcc", 0) for m in metrics]
    rmse_vals = [results[m].get("rmse", 0) for m in metrics]
    src_vals = [results[m].get("src", 0) for m in metrics]

    x = np.arange(len(metrics))
    w = 0.25
    fig, ax = plt.subplots(figsize=(max(8, len(metrics) * 2), 6))
    bars1 = ax.bar(x - w, pcc_vals, w, label="PCC", color="#818cf8")
    bars2 = ax.bar(x, rmse_vals, w, label="RMSE", color="#f97316")
    bars3 = ax.bar(x + w, src_vals, w, label="SRC", color="#34d399")
    ax.bar_label(bars1, fmt='%.3f', padding=3, fontsize=8, color=TEXT_COLOR)
    ax.bar_label(bars2, fmt='%.3f', padding=3, fontsize=8, color=TEXT_COLOR)
    ax.bar_label(bars3, fmt='%.3f', padding=3, fontsize=8, color=TEXT_COLOR)
    ax.set_xticks(x)
    ax.set_xticklabels(metrics, rotation=30, ha="right", fontsize=9)
    ax.set_title(f"Evaluation Metrics — {model_name} [{split}]\n(PCC/SRC: higher is better, RMSE: lower is better)",
                 fontsize=14, fontweight="bold")
    ax.legend()
    fig.tight_layout()
    model_type = model_name.split('_')[0]
    _save(fig, output_dir / split / model_type / f"metrics_bar_{model_name}_{split}.png")


# ═══════════════════════════════════════════════════════════════════════
# 3. Model comparison bar chart
# ═══════════════════════════════════════════════════════════════════════

def plot_model_comparison(
    split_results: dict, split: str, output_dir: Path,
):
    if not split_results:
        return
    model_names = list(split_results.keys())
    all_metrics = set()
    for res in split_results.values():
        all_metrics.update(res.keys())
    metrics = sorted(all_metrics)
    if not metrics:
        return

    x = np.arange(len(metrics))
    w = 0.8 / max(len(model_names), 1)
    fig, ax = plt.subplots(figsize=(max(10, len(metrics) * 2.5), 6))

    for i, mname in enumerate(model_names):
        vals = [split_results[mname].get(m, {}).get("pcc", 0) for m in metrics]
        bars = ax.bar(x + i * w - 0.4 + w / 2, vals, w,
               label=mname, color=_model_color(mname, i), edgecolor=CARD_BG)
        ax.bar_label(bars, fmt='%.3f', padding=3, fontsize=8, color=TEXT_COLOR)

    ax.set_xticks(x)
    ax.set_xticklabels(metrics, rotation=30, ha="right", fontsize=9)
    ax.set_ylabel("PCC")
    ax.set_title(f"Model Comparison (PCC) [{split}]\n(higher is better)", fontsize=14, fontweight="bold")
    ax.legend(fontsize=8, loc="lower right")
    fig.tight_layout()
    _save(fig, output_dir / split / f"model_comparison_{split}.png")


# ═══════════════════════════════════════════════════════════════════════
# 4. BiGRU checkpoint comparison
# ═══════════════════════════════════════════════════════════════════════

def plot_bigru_comparison(
    split_results: dict, split: str, output_dir: Path,
):
    bigru_models = {k: v for k, v in split_results.items() if k.startswith("bigru")}
    if len(bigru_models) < 2:
        return  # nothing to compare
    plot_model_comparison(bigru_models, split, output_dir)
    # save with distinct name
    src = output_dir / split / f"model_comparison_{split}.png"
    dst = output_dir / split / f"bigru_comparison_{split}.png"
    if src.exists():
        import shutil
        shutil.copy2(src, dst)


# ═══════════════════════════════════════════════════════════════════════
# 5. Error distribution histograms
# ═══════════════════════════════════════════════════════════════════════

def plot_error_distribution(
    targets: dict, predictions: dict, model_name: str,
    split: str, metrics: list[str], output_dir: Path,
):
    avail = [m for m in metrics if m in targets and m in predictions]
    if not avail:
        return
    n = len(avail)
    cols = min(n, 3)
    rows = (n + cols - 1) // cols
    fig, axes = plt.subplots(rows, cols, figsize=(6 * cols, 5 * rows), squeeze=False)
    fig.suptitle(f"Prediction Error Distribution — {model_name} [{split}]\n(closer to zero is better)",
                 fontsize=16, fontweight="bold", y=1.06)

    for i, metric in enumerate(avail):
        ax = axes[i // cols][i % cols]
        errors = np.asarray(predictions[metric]).ravel() - np.asarray(targets[metric]).ravel()
        ax.hist(errors, bins=40, color=ACCENT, edgecolor=CARD_BG, alpha=0.85)
        mu, sigma = errors.mean(), errors.std()
        ax.axvline(mu, color="#f97316", linestyle="--", linewidth=1.5)
        ax.text(0.97, 0.95, f"μ={mu:.3f}\nσ={sigma:.3f}",
                transform=ax.transAxes, fontsize=9, va="top", ha="right",
                bbox=dict(boxstyle="round,pad=0.3", fc=BG_COLOR, ec=GRID_COLOR, alpha=0.85))
        ax.set_title(metric, fontsize=11, fontweight="bold")
        ax.set_xlabel("Error (pred − true)")

    for j in range(n, rows * cols):
        axes[j // cols][j % cols].set_visible(False)

    fig.tight_layout()
    model_type = model_name.split('_')[0]
    _save(fig, output_dir / split / model_type / f"error_distribution_{model_name}_{split}.png")


# ═══════════════════════════════════════════════════════════════════════
# 6. Prediction frequency bias
# ═══════════════════════════════════════════════════════════════════════

def plot_prediction_frequency_bias(
    targets: dict, predictions: dict, model_name: str,
    split: str, metrics: list[str], output_dir: Path,
):
    avail = [m for m in metrics if m in targets and m in predictions]
    if not avail:
        return
    n = len(avail)
    cols = min(n, 3)
    rows = (n + cols - 1) // cols
    fig, axes = plt.subplots(rows, cols, figsize=(6 * cols, 5 * rows), squeeze=False)
    fig.suptitle(f"Prediction Frequency Bias — {model_name} [{split}]\n(closer to zero is better)",
                 fontsize=16, fontweight="bold", y=1.06)

    for i, metric in enumerate(avail):
        ax = axes[i // cols][i % cols]
        yt = np.asarray(targets[metric]).ravel()
        yp = np.asarray(predictions[metric]).ravel()

        # Round to nearest integer for discrete binning
        yt_r = np.round(yt).astype(int)
        yp_r = np.round(yp).astype(int)
        all_vals = sorted(set(yt_r) | set(yp_r))

        n_total = len(yt_r)
        gt_pct = {v: np.sum(yt_r == v) / n_total * 100 for v in all_vals}
        pred_pct = {v: np.sum(yp_r == v) / n_total * 100 for v in all_vals}
        bias = {v: pred_pct.get(v, 0) - gt_pct.get(v, 0) for v in all_vals}

        colors = ["#34d399" if b >= 0 else "#f87171" for b in bias.values()]
        bars = ax.bar([str(v) for v in all_vals], list(bias.values()), color=colors,
               edgecolor=CARD_BG, linewidth=0.5)
        ax.bar_label(bars, fmt='%.1f', padding=3, fontsize=8, color=TEXT_COLOR)
        ax.axhline(0, color=TEXT_COLOR, linewidth=0.8, alpha=0.5)
        ax.set_title(metric, fontsize=11, fontweight="bold")
        ax.set_xlabel("Score Value")
        ax.set_ylabel("Bias (pp)")
        ax.tick_params(axis="x", labelsize=8)

    for j in range(n, rows * cols):
        axes[j // cols][j % cols].set_visible(False)

    fig.tight_layout()
    model_type = model_name.split('_')[0]
    _save(fig, output_dir / split / model_type / f"prediction_bias_{model_name}_{split}.png")


# ═══════════════════════════════════════════════════════════════════════
# 7. Correlation matrix
# ═══════════════════════════════════════════════════════════════════════

def plot_correlation_matrix(
    targets: dict, predictions: dict, model_name: str,
    split: str, output_dir: Path,
):
    metrics = sorted(set(targets.keys()) & set(predictions.keys()))
    if len(metrics) < 2:
        return
    corr = np.zeros((len(metrics), len(metrics)))
    for i, m1 in enumerate(metrics):
        for j, m2 in enumerate(metrics):
            try:
                corr[i, j], _ = pearsonr(
                    np.asarray(predictions[m1]).ravel(),
                    np.asarray(predictions[m2]).ravel(),
                )
            except Exception:
                corr[i, j] = float("nan")

    fig, ax = plt.subplots(figsize=(max(6, len(metrics) * 1.2), max(5, len(metrics))))
    cmap = sns.diverging_palette(260, 20, as_cmap=True)
    sns.heatmap(corr, annot=True, fmt=".2f", cmap=cmap, vmin=-1, vmax=1,
                xticklabels=metrics, yticklabels=metrics,
                linewidths=1, linecolor=BG_COLOR, ax=ax, square=True)
    ax.set_title(f"Prediction Correlation — {model_name} [{split}]",
                 fontsize=13, fontweight="bold")
    ax.tick_params(labelsize=8)
    fig.tight_layout()
    model_type = model_name.split('_')[0]
    _save(fig, output_dir / split / model_type / f"correlation_matrix_{model_name}_{split}.png")


# ═══════════════════════════════════════════════════════════════════════
# 8. Dataset demographic bias
# ═══════════════════════════════════════════════════════════════════════

def _classify_age(age) -> str:
    try:
        age = int(age)
    except (TypeError, ValueError):
        return "Unknown"
    if age <= 12:
        return "Child (≤12)"
    elif age <= 17:
        return "Adolescent"
    return "Adult (18+)"


def plot_dataset_demographic_bias(
    manifest_df: pd.DataFrame, split: str, output_dir: Path,
):
    if manifest_df is None or manifest_df.empty:
        return
    df = manifest_df.copy()
    df["age_group"] = df["age"].apply(_classify_age)
    score_cols = [c for c in ("accuracy", "completeness", "fluency", "prosodic") if c in df.columns]
    if not score_cols:
        return

    fig, axes = plt.subplots(1, 2, figsize=(16, 6))
    fig.suptitle(f"Dataset Demographic Bias [{split}]",
                 fontsize=16, fontweight="bold", y=1.03)

    # Age group
    known = df[df["age_group"] != "Unknown"]
    age_order = [g for g in ["Child (≤12)", "Adolescent", "Adult (18+)"]
                 if g in known["age_group"].values]
    if age_order and len(score_cols) > 0:
        melt = known.melt(id_vars=["age_group"], value_vars=score_cols,
                          var_name="Metric", value_name="Score")
        sns.boxplot(data=melt, x="Metric", y="Score", hue="age_group",
                    hue_order=age_order, palette="cool", ax=axes[0],
                    linewidth=0.8, fliersize=1.5)
        axes[0].set_title("By Age Group", fontsize=13, fontweight="bold")
        axes[0].legend(fontsize=8)
    else:
        axes[0].set_visible(False)

    # Gender
    gender_known = df[df["gender"].isin(["m", "f"])] if "gender" in df.columns else pd.DataFrame()
    if not gender_known.empty and score_cols:
        melt = gender_known.melt(id_vars=["gender"], value_vars=score_cols,
                                 var_name="Metric", value_name="Score")
        sns.boxplot(data=melt, x="Metric", y="Score", hue="gender",
                    palette={"m": "#60a5fa", "f": "#f472b6"}, ax=axes[1],
                    linewidth=0.8, fliersize=1.5)
        axes[1].set_title("By Gender", fontsize=13, fontweight="bold")
        axes[1].legend(fontsize=8)
    else:
        axes[1].set_visible(False)

    fig.tight_layout()
    _save(fig, output_dir / split / f"dataset_demographic_bias_{split}.png")


# ═══════════════════════════════════════════════════════════════════════
# 9. Model demographic bias
# ═══════════════════════════════════════════════════════════════════════

def plot_model_demographic_bias(
    targets: dict, predictions: dict, speaker_ids: list[str],
    manifest_df: pd.DataFrame, model_name: str, split: str, output_dir: Path,
):
    if manifest_df is None or manifest_df.empty:
        return

    # Only sentence-level metrics (1 per sentence, matching speaker_ids length)
    sent_metrics = [m for m in targets if m.startswith("sentence_")
                    and m in predictions
                    and len(np.asarray(targets[m]).ravel()) == len(speaker_ids)]
    if not sent_metrics:
        return

    # Build speaker → demographic lookup
    spk_demo = {}
    for _, row in manifest_df.iterrows():
        sid = str(row["speaker_id"])
        spk_demo[sid] = {
            "age_group": _classify_age(row.get("age")),
            "gender": row.get("gender", "unknown"),
        }

    groups = [spk_demo.get(s, {}).get("age_group", "Unknown") for s in speaker_ids]

    fig, axes = plt.subplots(1, len(sent_metrics), figsize=(6 * len(sent_metrics), 6),
                             squeeze=False)
    fig.suptitle(f"Model Demographic Bias — {model_name} [{split}]\n(PCC: higher is better, RMSE: lower is better)",
                 fontsize=16, fontweight="bold", y=1.06)

    for i, metric in enumerate(sent_metrics):
        ax = axes[0][i]
        yt = np.asarray(targets[metric]).ravel()
        yp = np.asarray(predictions[metric]).ravel()
        errors = np.abs(yp - yt)

        rows_list = []
        for g in sorted(set(groups)):
            if g == "Unknown":
                continue
            mask = np.array([gr == g for gr in groups])
            if mask.sum() == 0:
                continue
            rmse = np.sqrt(np.mean(errors[mask] ** 2))
            try:
                pcc, _ = pearsonr(yt[mask], yp[mask])
            except Exception:
                pcc = float("nan")
            rows_list.append({"Group": g, "RMSE": rmse, "PCC": pcc})

        if not rows_list:
            ax.set_visible(False)
            continue

        gdf = pd.DataFrame(rows_list)
        x = np.arange(len(gdf))
        w = 0.35
        bars1 = ax.bar(x - w / 2, gdf["PCC"], w, label="PCC", color="#818cf8")
        bars2 = ax.bar(x + w / 2, gdf["RMSE"], w, label="RMSE", color="#f97316")
        ax.bar_label(bars1, fmt='%.3f', padding=3, fontsize=8, color=TEXT_COLOR)
        ax.bar_label(bars2, fmt='%.3f', padding=3, fontsize=8, color=TEXT_COLOR)
        ax.set_xticks(x)
        ax.set_xticklabels(gdf["Group"], fontsize=9)
        ax.set_title(metric, fontsize=11, fontweight="bold")
        ax.legend(fontsize=8)

    fig.tight_layout()
    model_type = model_name.split('_')[0]
    _save(fig, output_dir / split / model_type / f"demographic_bias_{model_name}_{split}.png")


# ═══════════════════════════════════════════════════════════════════════
# 10. Cross-split model performance (Total, Train, Val, Test)
# ═══════════════════════════════════════════════════════════════════════

def plot_cross_split_performance(
    all_data: dict[str, dict[str, tuple]],
    model_name: str,
    output_dir: Path,
):
    # metrics available for this model
    metrics = set()
    for split in ["train", "val", "test"]:
        if split in all_data and model_name in all_data[split]:
            metrics.update(all_data[split][model_name][0].keys())
    metrics = sorted(list(metrics))
    if not metrics:
        return

    splits_to_plot = ["total", "train", "val", "test"]
    pcc_data = {s: [] for s in splits_to_plot}

    for metric in metrics:
        total_targets = []
        total_preds = []
        for split in ["train", "val", "test"]:
            if split in all_data and model_name in all_data[split]:
                targets, predictions, _ = all_data[split][model_name]
                if metric in targets and metric in predictions:
                    yt = np.asarray(targets[metric]).ravel()
                    yp = np.asarray(predictions[metric]).ravel()
                    total_targets.extend(yt)
                    total_preds.extend(yp)
                    
                    if len(np.unique(yt)) > 1 and len(np.unique(yp)) > 1:
                        pcc, _ = pearsonr(yt, yp)
                    else:
                        pcc = float('nan')
                    pcc_data[split].append(pcc)
                else:
                    pcc_data[split].append(0)
            else:
                pcc_data[split].append(0)
                
        if total_targets:
            if len(np.unique(total_targets)) > 1 and len(np.unique(total_preds)) > 1:
                pcc_total, _ = pearsonr(total_targets, total_preds)
            else:
                pcc_total = float('nan')
            pcc_data["total"].append(pcc_total)
        else:
            pcc_data["total"].append(0)

    x = np.arange(len(metrics))
    n_splits = len(splits_to_plot)
    w = 0.8 / n_splits
    fig, ax = plt.subplots(figsize=(max(10, len(metrics) * 2.5), 6))

    colors = {"total": "#f472b6", "train": "#6366f1", "val": "#f59e0b", "test": "#10b981"}
    
    for i, split in enumerate(splits_to_plot):
        # Convert nans to 0 for plotting or leave as nan
        vals = [v if not np.isnan(v) else 0 for v in pcc_data[split]]
        bars = ax.bar(x + i * w - 0.4 + w / 2, vals, w,
               label=split.capitalize(), color=colors[split], edgecolor=CARD_BG)
        ax.bar_label(bars, fmt='%.3f', padding=3, fontsize=8, color=TEXT_COLOR)

    ax.set_xticks(x)
    ax.set_xticklabels(metrics, rotation=30, ha="right", fontsize=9)
    ax.set_ylabel("PCC")
    ax.set_title(f"Cross-Split Performance (PCC) — {model_name}\n(higher is better)", fontsize=14, fontweight="bold")
    ax.legend(fontsize=8, loc="lower right")
    fig.tight_layout()
    
    model_type = model_name.split('_')[0]
    _save(fig, output_dir / "all" / model_type / f"cross_split_performance_{model_name}.png")


# ═══════════════════════════════════════════════════════════════════════
# Orchestrator
# ═══════════════════════════════════════════════════════════════════════

def generate_all_charts(
    all_results: dict[str, dict],
    all_data: dict[str, dict[str, tuple]],
    manifests: dict[str, pd.DataFrame],
    active_metrics: list[str],
    bigru_checkpoints: list[str],
    output_dir: Path,
):
    """Generate all charts for all splits."""
    output_dir = Path(output_dir)
    splits = ["train", "val", "test"]

    for split in splits:
        split_results = all_results.get(split, {})
        split_data = all_data.get(split, {})
        manifest_df = manifests.get(split)

        print(f"\n▸ Charts for [{split}]")

        # Dataset demographic bias (once per split)
        if manifest_df is not None:
            plot_dataset_demographic_bias(manifest_df, split, output_dir)

        # Per-model charts
        for model_name, (targets, predictions, speaker_ids) in split_data.items():
            results = split_results.get(model_name, {})

            plot_scatter_predictions(targets, predictions, model_name, split,
                                     active_metrics, output_dir)
            plot_metrics_bar(results, model_name, split, output_dir)
            plot_error_distribution(targets, predictions, model_name, split,
                                     active_metrics, output_dir)
            plot_prediction_frequency_bias(targets, predictions, model_name, split,
                                            active_metrics, output_dir)
            plot_correlation_matrix(targets, predictions, model_name, split,
                                     output_dir)
            if manifest_df is not None:
                plot_model_demographic_bias(targets, predictions, speaker_ids,
                                            manifest_df, model_name, split,
                                            output_dir)

        # Cross-model comparison
        if len(split_results) > 1:
            plot_model_comparison(split_results, split, output_dir)

        # BiGRU cross-checkpoint comparison
        bigru_results = {k: v for k, v in split_results.items()
                         if k.startswith("bigru")}
        if len(bigru_results) >= 2:
            plot_bigru_comparison(split_results, split, output_dir)

    # ── "all" charts: aggregate across splits ───────────────────────
    print(f"\n▸ Charts for [all]")
    # Combine results from all splits for a combined model comparison
    combined_results: dict[str, dict] = {}
    for split in splits:
        for model_name, results in all_results.get(split, {}).items():
            key = f"{model_name}"
            if key not in combined_results:
                combined_results[key] = {}
            for metric, scores in results.items():
                if metric not in combined_results[key]:
                    combined_results[key][metric] = {
                        "pcc": [], "rmse": [], "src": []
                    }
                for k in ("pcc", "rmse", "src"):
                    combined_results[key][metric][k].append(scores.get(k, 0))

    # Average across splits
    avg_results: dict[str, dict] = {}
    for model_name, metrics_dict in combined_results.items():
        avg_results[model_name] = {}
        for metric, vals in metrics_dict.items():
            avg_results[model_name][metric] = {
                k: float(np.nanmean(v)) for k, v in vals.items()
            }

    if len(avg_results) > 1:
        plot_model_comparison(avg_results, "all", output_dir)
        
    # Generate cross-split performance charts for each model
    all_models = set()
    for split in splits:
        if split in all_data:
            all_models.update(all_data[split].keys())
            
    for model_name in all_models:
        plot_cross_split_performance(all_data, model_name, output_dir)
