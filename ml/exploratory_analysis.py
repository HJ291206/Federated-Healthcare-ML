"""Descriptive statistics and figures for the BRFSS 2015 diabetes data.

Run from the repository root (after ml.data_validation):
    python -m ml.exploratory_analysis

Writes:
    results/feature_statistics.csv
    results/figures/target_distribution.png
    results/figures/feature_distributions_by_class.png
    results/figures/binary_feature_prevalence_by_class.png
    results/figures/feature_correlation.png
"""

from __future__ import annotations

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from ml.data_validation import (
    BALANCED_FILE, BINARY_FEATURES, CONTINUOUS_FEATURES, COUNT_FEATURES, FEATURES,
    MULTICLASS_FILE, MULTICLASS_TARGET, ORDINAL_FEATURES, PRIMARY_FILE, RESULTS_DIR,
    TARGET, load_csv,
)

FIG_DIR = RESULTS_DIR / "figures"
CLASS_COLORS = {0: "#4C72B0", 1: "#DD8452", 2: "#C44E52"}

# Meanings from the UCI ML Repository variable table for dataset 891 (CDC Diabetes Health
# Indicators), retrieved 2026-10-14. MentHlth/PhysHlth are documented there as "1-30 days",
# but the files contain 0, so 0 is reported as observed rather than as documented.
FEATURE_METADATA = {
    "HighBP": "High blood pressure (0 = no, 1 = yes)",
    "HighChol": "High cholesterol (0 = no, 1 = yes)",
    "CholCheck": "Cholesterol check in past 5 years (0 = no, 1 = yes)",
    "BMI": "Body Mass Index (integer)",
    "Smoker": "Smoked at least 100 cigarettes in entire life (0 = no, 1 = yes)",
    "Stroke": "Ever told had a stroke (0 = no, 1 = yes)",
    "HeartDiseaseorAttack": "Coronary heart disease or myocardial infarction (0 = no, 1 = yes)",
    "PhysActivity": "Physical activity in past 30 days, not including job (0 = no, 1 = yes)",
    "Fruits": "Consume fruit 1+ times per day (0 = no, 1 = yes)",
    "Veggies": "Consume vegetables 1+ times per day (0 = no, 1 = yes)",
    "HvyAlcoholConsump": "Heavy drinker: men >14, women >7 drinks/week (0 = no, 1 = yes)",
    "AnyHealthcare": "Any health care coverage (0 = no, 1 = yes)",
    "NoDocbcCost": "Could not see doctor because of cost in past 12 months (0 = no, 1 = yes)",
    "GenHlth": "Self-rated general health (1 = excellent ... 5 = poor)",
    "MentHlth": "Days of poor mental health in past 30 days (observed 0-30)",
    "PhysHlth": "Days of poor physical health in past 30 days (observed 0-30)",
    "DiffWalk": "Serious difficulty walking or climbing stairs (0 = no, 1 = yes)",
    "Sex": "Sex (0 = female, 1 = male)",
    "Age": "13-level age category _AGEG5YR (1 = 18-24, 9 = 60-64, 13 = 80+)",
    "Education": "Education level EDUCA (1 = never attended/kindergarten ... 6 = college graduate)",
    "Income": "Income scale INCOME2 (1 = <$10,000, 5 = <$35,000, 8 = $75,000+)",
}


def feature_group(col: str) -> str:
    if col in BINARY_FEATURES:
        return "binary"
    if col in ORDINAL_FEATURES:
        return "ordinal"
    if col in COUNT_FEATURES:
        return "count"
    if col in CONTINUOUS_FEATURES:
        return "continuous"
    raise KeyError(col)


def feature_statistics(df: pd.DataFrame) -> pd.DataFrame:
    y = df[TARGET]
    neg, pos = df[y == 0], df[y == 1]
    rows = []
    for col in FEATURES:
        x = df[col]
        pooled_sd = np.sqrt((neg[col].var() + pos[col].var()) / 2)
        rows.append({
            "feature": col,
            "group": feature_group(col),
            "meaning": FEATURE_METADATA[col],
            "count": int(x.count()),
            "n_unique": int(x.nunique()),
            "mean": x.mean(),
            "std": x.std(),
            "min": x.min(),
            "q25": x.quantile(0.25),
            "median": x.median(),
            "q75": x.quantile(0.75),
            "max": x.max(),
            "mean_no_diabetes": neg[col].mean(),
            "mean_diabetes": pos[col].mean(),
            "standardized_mean_difference": (pos[col].mean() - neg[col].mean()) / pooled_sd,
            "pearson_r_with_target": x.corr(y),
        })
    out = pd.DataFrame(rows)
    num = out.select_dtypes("number").columns.difference(["count", "n_unique"])
    out[num] = out[num].round(4)
    return out


def _bar(ax, counts: pd.Series, labels: dict[int, str], title: str) -> None:
    total = counts.sum()
    bars = ax.bar([labels[k] for k in counts.index], counts.to_numpy(),
                  color=[CLASS_COLORS[k] for k in counts.index])
    for bar, n in zip(bars, counts.to_numpy()):
        ax.annotate(f"{n:,}\n({n / total:.1%})", (bar.get_x() + bar.get_width() / 2, bar.get_height()),
                    ha="center", va="bottom", fontsize=9)
    ax.set_title(title, fontsize=10)
    ax.set_ylabel("Respondents")
    ax.set_ylim(0, counts.max() * 1.2)
    ax.yaxis.set_major_formatter(matplotlib.ticker.FuncFormatter(lambda v, _: f"{v:,.0f}"))
    ax.spines[["top", "right"]].set_visible(False)


def plot_target_distribution(primary: pd.DataFrame, balanced: pd.DataFrame, multi: pd.DataFrame) -> None:
    binary_labels = {0: "0: no diabetes\n(incl. prediabetes)", 1: "1: diabetes"}
    fig, axes = plt.subplots(1, 3, figsize=(15, 4.8))
    _bar(axes[0], primary[TARGET].value_counts().sort_index(), binary_labels,
         f"Primary: {PRIMARY_FILE}\n(used for modelling)")
    _bar(axes[1], balanced[TARGET].value_counts().sort_index(), binary_labels,
         f"Alternative: {BALANCED_FILE}\n(balanced subsample of primary)")
    _bar(axes[2], multi[MULTICLASS_TARGET].value_counts().sort_index(),
         {0: "0: no diabetes\n(or pregnancy only)", 1: "1: prediabetes", 2: "2: diabetes"},
         f"Alternative: {MULTICLASS_FILE}\n(same respondents as primary)")
    fig.suptitle("Target distribution, BRFSS 2015 diabetes health indicators (observed counts)")
    fig.tight_layout()
    fig.savefig(FIG_DIR / "target_distribution.png", dpi=150)
    plt.close(fig)


def plot_feature_distributions(df: pd.DataFrame) -> None:
    cols = CONTINUOUS_FEATURES + list(ORDINAL_FEATURES) + list(COUNT_FEATURES)
    fig, axes = plt.subplots(2, 4, figsize=(16, 7))
    for ax, col in zip(axes.flat, cols):
        values = np.sort(df[col].unique())
        if col == "BMI":
            bins = np.arange(10, 101, 2)
        else:
            bins = np.arange(values.min() - 0.5, values.max() + 1.5, 1)
        for cls, label in [(0, "no diabetes"), (1, "diabetes")]:
            ax.hist(df.loc[df[TARGET] == cls, col], bins=bins, density=True, alpha=0.55,
                    color=CLASS_COLORS[cls], label=label)
        ax.set_title(col, fontsize=10)
        ax.set_ylabel("Density within class")
        ax.spines[["top", "right"]].set_visible(False)
    axes.flat[-1].axis("off")
    axes.flat[0].legend()
    fig.suptitle("Non-binary features by class (normalized within class), primary dataset")
    fig.tight_layout()
    fig.savefig(FIG_DIR / "feature_distributions_by_class.png", dpi=150)
    plt.close(fig)


def plot_binary_prevalence(df: pd.DataFrame) -> None:
    prev = df.groupby(TARGET)[BINARY_FEATURES].mean().T
    prev = prev.sort_values(1)
    y = np.arange(len(prev))
    fig, ax = plt.subplots(figsize=(8, 6.5))
    ax.barh(y - 0.2, prev[0], height=0.4, color=CLASS_COLORS[0], label="no diabetes")
    ax.barh(y + 0.2, prev[1], height=0.4, color=CLASS_COLORS[1], label="diabetes")
    ax.set_yticks(y, prev.index)
    ax.set_xlabel("Share of respondents with value 1")
    ax.set_xlim(0, 1)
    ax.legend(loc="lower right")
    ax.set_title("Binary indicators by class, primary dataset")
    ax.spines[["top", "right"]].set_visible(False)
    fig.tight_layout()
    fig.savefig(FIG_DIR / "binary_feature_prevalence_by_class.png", dpi=150)
    plt.close(fig)


def plot_correlation(df: pd.DataFrame) -> None:
    corr = df[[TARGET] + FEATURES].corr()
    fig, ax = plt.subplots(figsize=(11, 9))
    im = ax.imshow(corr, cmap="RdBu_r", vmin=-1, vmax=1)
    ax.set_xticks(range(len(corr)), corr.columns, rotation=90, fontsize=8)
    ax.set_yticks(range(len(corr)), corr.columns, fontsize=8)
    for i in range(len(corr)):
        for j in range(len(corr)):
            ax.text(j, i, f"{corr.iat[i, j]:.2f}", ha="center", va="center", fontsize=5.5,
                    color="white" if abs(corr.iat[i, j]) > 0.5 else "black")
    fig.colorbar(im, ax=ax, shrink=0.8, label="Pearson r")
    ax.set_title("Pearson correlation, primary dataset (linear association only)")
    fig.tight_layout()
    fig.savefig(FIG_DIR / "feature_correlation.png", dpi=150)
    plt.close(fig)


def main() -> None:
    FIG_DIR.mkdir(parents=True, exist_ok=True)
    primary = load_csv(PRIMARY_FILE)
    balanced = load_csv(BALANCED_FILE)
    multi = load_csv(MULTICLASS_FILE)

    stats = feature_statistics(primary)
    stats.to_csv(RESULTS_DIR / "feature_statistics.csv", index=False)

    plot_target_distribution(primary, balanced, multi)
    plot_feature_distributions(primary)
    plot_binary_prevalence(primary)
    plot_correlation(primary)

    counts = primary[TARGET].value_counts().sort_index()
    print(f"Primary: {len(primary):,} rows; class 0 = {counts[0]:,}, class 1 = {counts[1]:,} "
          f"({counts[1] / counts.sum():.2%} positive, {counts[0] / counts[1]:.2f}:1)")
    view = stats[["feature", "group", "mean", "std", "min", "max", "mean_no_diabetes",
                  "mean_diabetes", "standardized_mean_difference", "pearson_r_with_target"]]
    print(view.sort_values("pearson_r_with_target", key=abs, ascending=False).to_string(index=False))
    print(f"\nWrote {RESULTS_DIR / 'feature_statistics.csv'} and figures in {FIG_DIR}")


if __name__ == "__main__":
    main()
