"""
src/analysis/score_distributions.py
====================================
Comprehensive analysis of score distributions in the Speechocean762 dataset.

Generates publication-quality charts covering:
  1. Sentence-level score distributions  (accuracy, completeness, fluency, prosodic, total)
  2. Word-level score distributions       (accuracy, stress, total)
  3. Phoneme-level accuracy distribution  (0.0 – 2.0)
  4. Correlation heatmap among sentence-level metrics
  5. Score distributions by demographic    (age group & gender)
  6. Per-split (train / val / test) comparison

All figures are saved to  ``outputs/analysis/``.
"""

import json
import sys
from collections import Counter
from pathlib import Path

import matplotlib
matplotlib.use("Agg")  # non-interactive backend

import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
import numpy as np
import pandas as pd
import seaborn as sns

# ── project root ────────────────────────────────────────────────────────
PROJECT_ROOT = Path(__file__).resolve().parents[2]
SCORES_PATH  = PROJECT_ROOT / "data" / "raw" / "resource" / "scores.json"
SPLITS_DIR   = PROJECT_ROOT / "data" / "splits"
RAW_DIR      = PROJECT_ROOT / "data" / "raw"
OUTPUT_DIR   = PROJECT_ROOT / "outputs" / "analysis"

# ── Default theme ──────────────────────────────────────────────────────
PALETTE = {
    "train": "#3b82f6",   # blue-500
    "val":   "#f59e0b",   # amber-500
    "test":  "#10b981",   # emerald-500
}

ACCENT      = "#3b82f6"
ACCENT_DARK = "#2563eb"
BG_COLOR    = "white"
CARD_BG     = "white"
TEXT_COLOR  = "black"
GRID_COLOR  = "#e5e7eb"
FONT_FAMILY = "DejaVu Sans"

# ── matplotlib global styling ───────────────────────────────────────────
plt.rcParams.update({
    "axes.grid":          True,
    "grid.color":         GRID_COLOR,
    "grid.alpha":         0.4,
    "font.family":        FONT_FAMILY,
    "font.size":          11,
    "savefig.bbox":       "tight",
    "savefig.dpi":        200,
})


# ═══════════════════════════════════════════════════════════════════════
# DATA LOADING
# ═══════════════════════════════════════════════════════════════════════

def load_scores() -> dict:
    with open(SCORES_PATH, "r", encoding="utf-8") as fh:
        return json.load(fh)


def load_splits() -> dict[str, set[str]]:
    """Return {split_name: {sentence_id, ...}} from the manifest CSVs."""
    splits: dict[str, set[str]] = {}
    for csv_path in sorted(SPLITS_DIR.glob("*_manifest.csv")):
        split_name = csv_path.stem.replace("_manifest", "")
        df = pd.read_csv(csv_path, dtype=str)
        splits[split_name] = set(df["sentence_id"].values)
    return splits


def load_demographics() -> dict[str, dict]:
    """Load speaker age and gender from the Kaldi-style metadata files."""
    demos: dict[str, dict] = {}
    for split_dir_name in ("train", "test"):
        split_dir = RAW_DIR / split_dir_name
        # ages
        age_file = split_dir / "spk2age"
        if age_file.exists():
            for line in age_file.read_text().strip().splitlines():
                parts = line.split()
                spk_id, age = parts[0], int(parts[1])
                demos.setdefault(spk_id, {})["age"] = age
        # gender
        gender_file = split_dir / "spk2gender"
        if gender_file.exists():
            for line in gender_file.read_text().strip().splitlines():
                parts = line.split()
                spk_id, gender = parts[0], parts[1]
                demos.setdefault(spk_id, {})["gender"] = gender
    return demos


def build_dataframe(scores: dict, splits: dict[str, set[str]], demos: dict) -> pd.DataFrame:
    """Build a flat DataFrame with one row per sentence, annotated with split and demographics."""
    rows = []
    # build utt2spk lookup
    utt2spk: dict[str, str] = {}
    for split_dir_name in ("train", "test"):
        utt2spk_file = RAW_DIR / split_dir_name / "utt2spk"
        if utt2spk_file.exists():
            for line in utt2spk_file.read_text().strip().splitlines():
                parts = line.split()
                utt2spk[parts[0]] = parts[1]

    for sid, entry in scores.items():
        # determine split
        split = "unknown"
        for s_name, s_ids in splits.items():
            if sid in s_ids:
                split = s_name
                break

        # demographics
        spk_id = utt2spk.get(sid, "")
        demo = demos.get(spk_id, {})
        age = demo.get("age", None)
        gender = demo.get("gender", None)
        age_group = _age_group(age)

        # word & phoneme stats
        words = entry.get("words", [])
        word_accs   = [w.get("accuracy", 0) for w in words]
        word_stress = [w.get("stress", 0) for w in words]
        word_totals = [w.get("total", 0)   for w in words]
        phone_accs  = []
        for w in words:
            phone_accs.extend(w.get("phones-accuracy", []))

        rows.append({
            "sentence_id":  sid,
            "split":        split,
            "speaker_id":   spk_id,
            "age":          age,
            "age_group":    age_group,
            "gender":       gender,
            "accuracy":     entry.get("accuracy", 0),
            "completeness": entry.get("completeness", 0),
            "fluency":      entry.get("fluency", 0),
            "prosodic":     entry.get("prosodic", 0),
            "total":        entry.get("total", 0),
            "n_words":      len(words),
            "n_phones":     len(phone_accs),
            "word_acc_mean":    np.mean(word_accs)   if word_accs else 0,
            "word_stress_mean": np.mean(word_stress) if word_stress else 0,
            "word_total_mean":  np.mean(word_totals) if word_totals else 0,
            "phone_acc_mean":   np.mean(phone_accs)  if phone_accs else 0,
        })

    return pd.DataFrame(rows)


def _age_group(age) -> str:
    if age is None:
        return "Unknown"
    if age <= 12:
        return "Child (≤12)"
    elif age <= 17:
        return "Adolescent (13-17)"
    else:
        return "Adult (18+)"


def build_word_df(scores: dict, splits: dict[str, set[str]]) -> pd.DataFrame:
    """One row per word."""
    rows = []
    for sid, entry in scores.items():
        split = "unknown"
        for s_name, s_ids in splits.items():
            if sid in s_ids:
                split = s_name
                break
        for w in entry.get("words", []):
            rows.append({
                "sentence_id": sid,
                "split":       split,
                "accuracy":    w.get("accuracy", 0),
                "stress":      w.get("stress", 0),
                "total":       w.get("total", 0),
                "text":        w.get("text", ""),
            })
    return pd.DataFrame(rows)


def build_phone_df(scores: dict, splits: dict[str, set[str]]) -> pd.DataFrame:
    """One row per phoneme."""
    rows = []
    for sid, entry in scores.items():
        split = "unknown"
        for s_name, s_ids in splits.items():
            if sid in s_ids:
                split = s_name
                break
        for w in entry.get("words", []):
            phones = w.get("phones", [])
            accs   = w.get("phones-accuracy", [])
            for phone, acc in zip(phones, accs):
                rows.append({
                    "sentence_id": sid,
                    "split":       split,
                    "phone":       phone,
                    "accuracy":    acc,
                })
    return pd.DataFrame(rows)


# ═══════════════════════════════════════════════════════════════════════
# CHART GENERATORS
# ═══════════════════════════════════════════════════════════════════════

def _save(fig, name: str):
    path = OUTPUT_DIR / f"{name}.png"
    fig.savefig(path)
    plt.close(fig)
    print(f"  ✓ saved {path.relative_to(PROJECT_ROOT)}")


# ── 1. Sentence-level histograms ────────────────────────────────────

def plot_sentence_distributions(df: pd.DataFrame):
    """5 histograms for sentence-level scores, one per metric."""
    metrics = ["accuracy", "fluency", "prosodic", "completeness", "total"]
    fig, axes = plt.subplots(1, 5, figsize=(26, 5))
    fig.suptitle("Sentence-Level Score Distributions", fontsize=18, fontweight="bold",
                 color=TEXT_COLOR, y=1.04)

    for ax, metric in zip(axes, metrics):
        data = df[metric].dropna()
        bins = np.arange(data.min() - 0.5, data.max() + 1.5, 1) if metric != "completeness" \
               else np.linspace(0, 1.05, 22)

        counts, edges = np.histogram(data, bins=bins)
        centres = 0.5 * (edges[:-1] + edges[1:])
        colors = [plt.cm.cool(v / max(counts.max(), 1)) for v in counts]

        ax.bar(centres, counts, width=(edges[1] - edges[0]) * 0.85, color=colors,
               edgecolor=CARD_BG, linewidth=0.5)

        # stats annotation
        stats_text = f"μ = {data.mean():.2f}\nσ = {data.std():.2f}\nmed = {data.median():.1f}"
        ax.text(0.97, 0.97, stats_text, transform=ax.transAxes, fontsize=9,
                va="top", ha="right",
                bbox=dict(boxstyle="round,pad=0.4", fc=BG_COLOR, ec=GRID_COLOR, alpha=0.85))

        ax.set_title(metric.capitalize(), fontsize=13, fontweight="bold", pad=8)
        ax.set_xlabel("Score")
        ax.set_ylabel("Count" if metric == metrics[0] else "")
        ax.yaxis.set_major_formatter(mticker.FuncFormatter(lambda x, _: f"{int(x):,}"))

    fig.tight_layout()
    _save(fig, "01_sentence_level_distributions")


# ── 2. Word-level histograms ────────────────────────────────────────

def plot_word_distributions(word_df: pd.DataFrame):
    """3 histograms for word-level accuracy, stress, total."""
    metrics = ["accuracy", "stress", "total"]
    fig, axes = plt.subplots(1, 3, figsize=(18, 5.5))
    fig.suptitle("Word-Level Score Distributions", fontsize=18, fontweight="bold",
                 color=TEXT_COLOR, y=1.04)

    for ax, metric in zip(axes, metrics):
        data = word_df[metric].dropna()
        bins = np.arange(-0.5, 11.5, 1)
        counts, edges = np.histogram(data, bins=bins)
        centres = 0.5 * (edges[:-1] + edges[1:])
        colors = [plt.cm.viridis(v / max(counts.max(), 1)) for v in counts]

        ax.bar(centres, counts, width=0.8, color=colors,
               edgecolor=CARD_BG, linewidth=0.5)

        stats_text = f"μ = {data.mean():.2f}\nσ = {data.std():.2f}\nmed = {data.median():.1f}"
        ax.text(0.03, 0.97, stats_text, transform=ax.transAxes, fontsize=9,
                va="top", ha="left",
                bbox=dict(boxstyle="round,pad=0.4", fc=BG_COLOR, ec=GRID_COLOR, alpha=0.85))

        ax.set_title(metric.capitalize(), fontsize=13, fontweight="bold", pad=8)
        ax.set_xlabel("Score")
        ax.set_ylabel("Count" if metric == metrics[0] else "")
        ax.set_xticks(range(0, 11))
        ax.yaxis.set_major_formatter(mticker.FuncFormatter(lambda x, _: f"{int(x):,}"))

    fig.tight_layout()
    _save(fig, "02_word_level_distributions")


# ── 3. Phoneme-level accuracy ───────────────────────────────────────

def plot_phoneme_distribution(phone_df: pd.DataFrame):
    """Histogram of phoneme-level accuracy (0.0 – 2.0)."""
    fig, ax = plt.subplots(figsize=(10, 5.5))
    fig.suptitle("Phoneme-Level Accuracy Distribution", fontsize=18,
                 fontweight="bold", color=TEXT_COLOR, y=1.02)

    data = phone_df["accuracy"].dropna()
    bins = np.arange(-0.05, 2.15, 0.1)
    counts, edges = np.histogram(data, bins=bins)
    centres = 0.5 * (edges[:-1] + edges[1:])
    colors = [plt.cm.magma(0.3 + 0.6 * v / max(counts.max(), 1)) for v in counts]

    ax.bar(centres, counts, width=0.08, color=colors, edgecolor=CARD_BG, linewidth=0.5)

    stats_text = (f"N = {len(data):,}\nμ = {data.mean():.3f}\nσ = {data.std():.3f}"
                  f"\nmed = {data.median():.2f}")
    ax.text(0.03, 0.97, stats_text, transform=ax.transAxes, fontsize=10,
            va="top", ha="left",
            bbox=dict(boxstyle="round,pad=0.4", fc=BG_COLOR, ec=GRID_COLOR, alpha=0.85))

    ax.set_xlabel("Phoneme Accuracy Score")
    ax.set_ylabel("Count")
    ax.set_xticks(np.arange(0, 2.1, 0.2))
    ax.yaxis.set_major_formatter(mticker.FuncFormatter(lambda x, _: f"{int(x):,}"))

    fig.tight_layout()
    _save(fig, "03_phoneme_level_distribution")


# ── 4. Correlation heatmap ──────────────────────────────────────────

def plot_correlation_heatmap(df: pd.DataFrame):
    """Pearson correlation among sentence-level metrics."""
    metrics = ["accuracy", "completeness", "fluency", "prosodic", "total"]
    corr = df[metrics].corr()

    fig, ax = plt.subplots(figsize=(8, 6.5))
    fig.suptitle("Sentence-Level Score Correlation Matrix", fontsize=16,
                 fontweight="bold", color=TEXT_COLOR, y=1.02)

    mask = np.triu(np.ones_like(corr, dtype=bool), k=1)
    cmap = sns.diverging_palette(260, 20, as_cmap=True)
    sns.heatmap(corr, mask=mask, annot=True, fmt=".2f", cmap=cmap,
                vmin=0, vmax=1, linewidths=1.5, linecolor=BG_COLOR,
                cbar_kws={"shrink": 0.8, "label": "Pearson r"},
                ax=ax, square=True)
    ax.set_xticklabels([m.capitalize() for m in metrics], rotation=30, ha="right")
    ax.set_yticklabels([m.capitalize() for m in metrics], rotation=0)

    fig.tight_layout()
    _save(fig, "04_correlation_heatmap")


# ── 5. Per-split comparison (violin) ────────────────────────────────

def plot_split_comparison(df: pd.DataFrame):
    """Violin + box plots comparing train/val/test for each sentence metric."""
    metrics = ["accuracy", "fluency", "prosodic", "total"]
    fig, axes = plt.subplots(1, 4, figsize=(22, 5.5))
    fig.suptitle("Score Distributions by Data Split", fontsize=18,
                 fontweight="bold", color=TEXT_COLOR, y=1.04)

    split_order = ["train", "val", "test"]
    for ax, metric in zip(axes, metrics):
        plot_df = df[df["split"].isin(split_order)]
        sns.violinplot(data=plot_df, x="split", y=metric, order=split_order,
                       palette=PALETTE, inner="box", linewidth=1, ax=ax,
                       saturation=0.9, cut=0)
        ax.set_title(metric.capitalize(), fontsize=13, fontweight="bold", pad=8)
        ax.set_xlabel("")
        ax.set_ylabel("Score" if metric == metrics[0] else "")

    fig.tight_layout()
    _save(fig, "05_split_comparison_violins")


# ── 6. Demographics: age group ──────────────────────────────────────

def plot_age_group_distributions(df: pd.DataFrame):
    """Box plots of sentence-level total by age group, split by gender."""
    fig, axes = plt.subplots(1, 2, figsize=(16, 6))
    fig.suptitle("Sentence Total Score by Demographics", fontsize=18,
                 fontweight="bold", color=TEXT_COLOR, y=1.04)

    known = df[df["age_group"] != "Unknown"].copy()
    age_order = ["Child (≤12)", "Adolescent (13-17)", "Adult (18+)"]

    # -- age group
    ax = axes[0]
    sns.boxplot(data=known, x="age_group", y="total", order=age_order,
                palette="cool", linewidth=1, fliersize=2, ax=ax, saturation=0.8)
    ax.set_title("By Age Group", fontsize=13, fontweight="bold", pad=8)
    ax.set_xlabel("")
    ax.set_ylabel("Total Score")

    # -- gender
    ax = axes[1]
    gender_known = known[known["gender"].isin(["m", "f"])]
    sns.boxplot(data=gender_known, x="gender", y="total",
                palette={"m": "#60a5fa", "f": "#f472b6"},
                linewidth=1, fliersize=2, ax=ax, saturation=0.8)
    ax.set_title("By Gender", fontsize=13, fontweight="bold", pad=8)
    ax.set_xlabel("")
    ax.set_ylabel("")
    ax.set_xticklabels(["Male", "Female"])

    fig.tight_layout()
    _save(fig, "06_demographic_distributions")


# ── 7. Top-10 most-frequent phonemes with accuracy box plots ────────

def plot_top_phonemes(phone_df: pd.DataFrame):
    """Box plots of accuracy for the 10 most frequent phonemes."""
    top_phones = phone_df["phone"].value_counts().head(10).index.tolist()
    subset = phone_df[phone_df["phone"].isin(top_phones)]

    fig, ax = plt.subplots(figsize=(14, 6))
    fig.suptitle("Accuracy Distribution for Top-10 Phonemes", fontsize=16,
                 fontweight="bold", color=TEXT_COLOR, y=1.02)

    sns.boxplot(data=subset, x="phone", y="accuracy", order=top_phones,
                palette="cool", linewidth=1, fliersize=1.5, ax=ax, saturation=0.8)
    ax.set_xlabel("Phoneme")
    ax.set_ylabel("Accuracy Score")
    ax.set_ylim(-0.1, 2.15)

    # add count annotation
    for i, ph in enumerate(top_phones):
        n = (phone_df["phone"] == ph).sum()
        ax.text(i, -0.07, f"n={n:,}", ha="center", va="top", fontsize=8, color=TEXT_COLOR)

    fig.tight_layout()
    _save(fig, "07_top_phoneme_accuracy")


# ── 8. Sentence length vs total score ───────────────────────────────

def plot_length_vs_score(df: pd.DataFrame):
    """Scatter plot of sentence length (n_words) vs total score."""
    fig, axes = plt.subplots(1, 2, figsize=(16, 6))
    fig.suptitle("Sentence Complexity vs. Scores", fontsize=18,
                 fontweight="bold", color=TEXT_COLOR, y=1.04)

    # words vs total
    ax = axes[0]
    ax.scatter(df["n_words"], df["total"], alpha=0.15, s=10, c=ACCENT, edgecolors="none")
    z = np.polyfit(df["n_words"], df["total"], 1)
    p = np.poly1d(z)
    x_range = np.linspace(df["n_words"].min(), df["n_words"].max(), 100)
    ax.plot(x_range, p(x_range), color="#f97316", linewidth=2, label=f"trend (r={df['n_words'].corr(df['total']):.2f})")
    ax.set_xlabel("Number of Words")
    ax.set_ylabel("Total Score")
    ax.set_title("Words per Sentence", fontsize=13, fontweight="bold", pad=8)
    ax.legend(loc="lower right", fontsize=9)

    # phones vs total
    ax = axes[1]
    ax.scatter(df["n_phones"], df["total"], alpha=0.15, s=10, c="#a78bfa", edgecolors="none")
    z = np.polyfit(df["n_phones"], df["total"], 1)
    p = np.poly1d(z)
    x_range = np.linspace(df["n_phones"].min(), df["n_phones"].max(), 100)
    ax.plot(x_range, p(x_range), color="#f97316", linewidth=2, label=f"trend (r={df['n_phones'].corr(df['total']):.2f})")
    ax.set_xlabel("Number of Phonemes")
    ax.set_ylabel("")
    ax.set_title("Phonemes per Sentence", fontsize=13, fontweight="bold", pad=8)
    ax.legend(loc="lower right", fontsize=9)

    fig.tight_layout()
    _save(fig, "08_length_vs_score")


# ── 9. Summary statistics table ─────────────────────────────────────

def generate_summary_table(df: pd.DataFrame, word_df: pd.DataFrame,
                           phone_df: pd.DataFrame) -> pd.DataFrame:
    """Generate and print a summary statistics table."""
    summary_rows = []

    # Sentence-level
    for metric in ["accuracy", "completeness", "fluency", "prosodic", "total"]:
        data = df[metric].dropna()
        summary_rows.append({
            "Level":    "Sentence",
            "Metric":   metric.capitalize(),
            "Count":    len(data),
            "Mean":     round(data.mean(), 2),
            "Std":      round(data.std(), 2),
            "Min":      data.min(),
            "25%":      round(data.quantile(0.25), 2),
            "Median":   round(data.median(), 2),
            "75%":      round(data.quantile(0.75), 2),
            "Max":      data.max(),
        })

    # Word-level
    for metric in ["accuracy", "stress", "total"]:
        data = word_df[metric].dropna()
        summary_rows.append({
            "Level":    "Word",
            "Metric":   metric.capitalize(),
            "Count":    len(data),
            "Mean":     round(data.mean(), 2),
            "Std":      round(data.std(), 2),
            "Min":      data.min(),
            "25%":      round(data.quantile(0.25), 2),
            "Median":   round(data.median(), 2),
            "75%":      round(data.quantile(0.75), 2),
            "Max":      data.max(),
        })

    # Phoneme-level
    data = phone_df["accuracy"].dropna()
    summary_rows.append({
        "Level":    "Phoneme",
        "Metric":   "Accuracy",
        "Count":    len(data),
        "Mean":     round(data.mean(), 3),
        "Std":      round(data.std(), 3),
        "Min":      data.min(),
        "25%":      round(data.quantile(0.25), 2),
        "Median":   round(data.median(), 2),
        "75%":      round(data.quantile(0.75), 2),
        "Max":      data.max(),
    })

    summary = pd.DataFrame(summary_rows)
    return summary


# ═══════════════════════════════════════════════════════════════════════
# MAIN
# ═══════════════════════════════════════════════════════════════════════

def main():
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    print("═" * 60)
    print("  Speechocean762 — Score Distribution Analysis")
    print("═" * 60)

    print("\n▸ Loading data …")
    scores = load_scores()
    splits = load_splits()
    demos  = load_demographics()
    print(f"  {len(scores):,} sentences loaded")
    for s, ids in sorted(splits.items()):
        print(f"  {s:>6s}: {len(ids):,} sentences")

    print("\n▸ Building DataFrames …")
    df       = build_dataframe(scores, splits, demos)
    word_df  = build_word_df(scores, splits)
    phone_df = build_phone_df(scores, splits)
    print(f"  sentences : {len(df):,}")
    print(f"  words     : {len(word_df):,}")
    print(f"  phonemes  : {len(phone_df):,}")

    print("\n▸ Generating charts …")
    plot_sentence_distributions(df)
    plot_word_distributions(word_df)
    plot_phoneme_distribution(phone_df)
    plot_correlation_heatmap(df)
    plot_split_comparison(df)
    plot_age_group_distributions(df)
    plot_top_phonemes(phone_df)
    plot_length_vs_score(df)

    print("\n▸ Summary statistics")
    summary = generate_summary_table(df, word_df, phone_df)
    print(summary.to_string(index=False))
    summary.to_csv(OUTPUT_DIR / "summary_statistics.csv", index=False)
    print(f"\n  ✓ saved {(OUTPUT_DIR / 'summary_statistics.csv').relative_to(PROJECT_ROOT)}")

    print("\n" + "═" * 60)
    print(f"  All outputs written to  {OUTPUT_DIR.relative_to(PROJECT_ROOT)}/")
    print("═" * 60)


if __name__ == "__main__":
    main()
