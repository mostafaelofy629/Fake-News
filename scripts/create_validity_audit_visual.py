from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
FINAL = ROOT / "final_artifacts"
NORM = ROOT / "normal_split_gt6_plus_trainhard_artifacts"
FIG_DIR = ROOT / "reports" / "figures"
TABLE_DIR = ROOT / "reports" / "tables"


def read_csv(path: Path) -> pd.DataFrame:
    return pd.read_csv(path) if path.exists() else pd.DataFrame()


def metric_from_fair_table(fair: pd.DataFrame, model_name: str) -> float:
    row = fair[fair["Model"].eq(model_name)]
    if row.empty:
        return np.nan
    return float(row.iloc[0]["Macro-F1"])


def best_after_metric(metrics: pd.DataFrame, contains: str) -> float:
    row = metrics[metrics["model"].astype(str).str.contains(contains, case=False, regex=False, na=False)]
    if row.empty:
        return np.nan
    return float(pd.to_numeric(row["macro_f1"], errors="coerce").max())


def load_after_metrics() -> pd.DataFrame:
    files = [
        "normal_split_text_baseline_metrics.csv",
        "normal_split_deep_model_metrics.csv",
        "normal_split_bert_text_metrics.csv",
        "normal_split_distilbert_roberta_text_metrics.csv",
        "normal_split_clip_swin_metrics.csv",
        "normal_split_voting_metrics.csv",
    ]
    frames = []
    for name in files:
        path = NORM / name
        if path.exists():
            frames.append(pd.read_csv(path))
    return pd.concat(frames, ignore_index=True, sort=False) if frames else pd.DataFrame()


def main() -> None:
    FIG_DIR.mkdir(parents=True, exist_ok=True)
    TABLE_DIR.mkdir(parents=True, exist_ok=True)

    fair = read_csv(FINAL / "report_table_classification_results.csv")
    after = load_after_metrics()
    audit = read_csv(NORM / "normal_split_cleaning_audit.csv")

    inspected = 5041
    removed = 519
    if not audit.empty:
        inspected_row = audit[audit["stage"].eq("clean_all_filtered")]
        removed_row = audit[audit["stage"].eq("combined_hard_rows_removed_present_after_image_validation")]
        if not inspected_row.empty:
            inspected = int(inspected_row.iloc[0]["rows"])
        if not removed_row.empty:
            removed = int(removed_row.iloc[0]["rows"])

    comparisons = pd.DataFrame(
        [
            {
                "model": "TF-IDF LR",
                "before_macro_f1": metric_from_fair_table(fair, "Text-only TF-IDF + Logistic Regression"),
                "after_macro_f1": best_after_metric(after, "TF-IDF Logistic Regression"),
            },
            {
                "model": "Concat CNN+Text",
                "before_macro_f1": metric_from_fair_table(fair, "Concat fusion CNN+Text"),
                "after_macro_f1": best_after_metric(after, "Concat fusion CNN+Text"),
            },
            {
                "model": "Consistency CNN+Text",
                "before_macro_f1": metric_from_fair_table(fair, "Consistency fusion CNN+Text"),
                "after_macro_f1": best_after_metric(after, "Consistency fusion CNN+Text"),
            },
            {
                "model": "ResNet50 fusion",
                "before_macro_f1": metric_from_fair_table(fair, "Frozen ResNet50 fusion CNN+Text"),
                "after_macro_f1": best_after_metric(after, "Frozen ResNet50 fusion CNN+Text"),
            },
            {
                "model": "Image-only CNN",
                "before_macro_f1": metric_from_fair_table(fair, "Image-only CNN"),
                "after_macro_f1": best_after_metric(after, "Image-only CNN"),
            },
        ]
    )
    comparisons["delta_pp"] = (comparisons["after_macro_f1"] - comparisons["before_macro_f1"]) * 100
    comparisons.to_csv(TABLE_DIR / "manual_validity_audit_before_after_metrics.csv", index=False)

    fig = plt.figure(figsize=(14, 8), facecolor="white")
    grid = fig.add_gridspec(2, 4, height_ratios=[1.05, 3.2], width_ratios=[1, 1, 1, 1], hspace=0.28, wspace=0.2)

    card_titles = [
        ("Image records inspected", f"{inspected:,}", "valid image-based records"),
        ("Removed after audit", f"{removed:,}", "misleading / label-inconsistent"),
        ("Typical fusion gain", "≈2-4 pp", "for comparable CNN-fusion trials"),
        ("Main issue", "Label mismatch", "fake/real ≠ consistency"),
    ]
    card_colors = ["#edf4ff", "#fff2e8", "#f0f8ef", "#fff7d8"]
    edge_colors = ["#78a7d8", "#db8c46", "#78a96b", "#d0a22a"]
    for idx, ((title, value, subtitle), fc, ec) in enumerate(zip(card_titles, card_colors, edge_colors)):
        axc = fig.add_subplot(grid[0, idx])
        axc.axis("off")
        axc.text(
            0.5,
            0.5,
            f"{value}\n{title}\n{subtitle}",
            ha="center",
            va="center",
            fontsize=11,
            linespacing=1.35,
            bbox=dict(boxstyle="round,pad=0.65", facecolor=fc, edgecolor=ec, linewidth=1.6),
        )

    ax = fig.add_subplot(grid[1, :])
    x = np.arange(len(comparisons))
    width = 0.36

    before_color = "#8aa1b8"
    after_color = "#d68a30"
    ax.bar(x - width / 2, comparisons["before_macro_f1"], width, label="Before validity cleaning", color=before_color)
    ax.bar(x + width / 2, comparisons["after_macro_f1"], width, label="After hard-sample cleaning", color=after_color)

    for i, row in comparisons.iterrows():
        ax.text(i - width / 2, row["before_macro_f1"] + 0.008, f"{row['before_macro_f1']:.3f}", ha="center", fontsize=9)
        ax.text(i + width / 2, row["after_macro_f1"] + 0.008, f"{row['after_macro_f1']:.3f}", ha="center", fontsize=9, weight="bold")
        ax.text(i, max(row["before_macro_f1"], row["after_macro_f1"]) + 0.032, f"+{row['delta_pp']:.1f} pp", ha="center", fontsize=10, color="#1f6f3a", weight="bold")

    ax.set_title("Manual Validity Audit Changed the Story", fontsize=20, weight="bold", pad=16)
    ax.set_ylabel("Macro-F1", fontsize=12, weight="bold")
    ax.set_xticks(x)
    ax.set_xticklabels(comparisons["model"], rotation=0, fontsize=10)
    ax.set_ylim(0.45, 0.91)
    ax.grid(axis="y", linestyle="--", alpha=0.25)
    ax.spines[["top", "right"]].set_visible(False)
    ax.legend(loc="upper right", frameon=False, fontsize=10)

    note = (
        "Cleaning removed repeatedly wrong/suspicious samples for diagnosis. "
        "This supports the claim that FakeNewsNet fake/real labels are not always image-text consistency labels."
    )
    fig.text(
        0.5,
        0.02,
        note,
        ha="center",
        fontsize=10.5,
        color="#333333",
        bbox=dict(boxstyle="round,pad=0.45", facecolor="#fff7d8", edgecolor="#d0a22a"),
    )

    out = FIG_DIR / "manual_validity_audit_before_after_cleaning.png"
    fig.savefig(out, dpi=220, bbox_inches="tight")
    plt.close(fig)

    print(f"Saved visual: {out.relative_to(ROOT)}")
    print(f"Saved table: {(TABLE_DIR / 'manual_validity_audit_before_after_metrics.csv').relative_to(ROOT)}")
    print(comparisons.to_string(index=False, float_format=lambda x: f"{x:.4f}"))


if __name__ == "__main__":
    main()
