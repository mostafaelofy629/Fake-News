from __future__ import annotations

from pathlib import Path
import textwrap

import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle
import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
ART = ROOT / "normal_split_gt6_plus_trainhard_artifacts"
FIG_DIR = ROOT / "reports" / "figures"
TABLE_DIR = ROOT / "reports" / "tables"


def compact_name(name: str) -> str:
    name = str(name).replace("NormalSplit hard-removed ", "")
    replacements = {
        "TF-IDF Logistic Regression": "TF-IDF LR",
        "BERT text classifier": "BERT",
        "DistilBERT text-only": "DistilBERT",
        "RoBERTa text-only": "RoBERTa",
        "soft voting ensemble": "Soft voting",
        "hard voting ensemble": "Hard voting",
        "CLIP ViT-B/32 frozen multimodal fusion": "CLIP ViT-B/32 fusion",
        "Swin-T frozen image + TF-IDF fusion": "Swin-T + TF-IDF fusion",
        "OCR+BLIP+BERT+ResNet50 cached-feature MLP": "OCR+BLIP+BERT+ResNet50 MLP",
        "Frozen ConvNeXt-Tiny fusion CNN+Text": "ConvNeXt-Tiny fusion",
        "Frozen ResNet50 fusion CNN+Text": "ResNet50 fusion",
        "Concat fusion CNN+Text": "Concat CNN+Text",
        "Consistency fusion CNN+Text": "Consistency CNN+Text",
        "Image-only CNN": "Image-only CNN",
        "Majority baseline": "Majority",
    }
    for old, new in replacements.items():
        name = name.replace(old, new)
    name = name.replace(" seed", " s")
    return name


def model_family(name: str) -> str:
    lower = str(name).lower()
    if "majority" in lower:
        return "majority"
    if any(x in lower for x in ["bert", "roberta", "tf-idf"]) and not any(x in lower for x in ["ocr", "blip", "resnet", "clip", "swin"]):
        return "text-only"
    if "image-only" in lower:
        return "image-only"
    if "voting" in lower:
        return "ensemble"
    if any(x in lower for x in ["clip", "swin", "fusion", "resnet", "convnext", "ocr", "blip"]):
        return "multimodal"
    return "other"


def load_metrics() -> pd.DataFrame:
    metric_files = [
        ART / "normal_split_text_baseline_metrics.csv",
        ART / "normal_split_bert_text_metrics.csv",
        ART / "normal_split_distilbert_roberta_text_metrics.csv",
        ART / "normal_split_deep_model_metrics.csv",
        ART / "normal_split_clip_swin_metrics.csv",
        ART / "normal_split_rich_cached_mlp_metrics.csv",
        ART / "normal_split_voting_metrics.csv",
    ]
    frames = []
    for path in metric_files:
        if path.exists():
            df = pd.read_csv(path)
            df["source_file"] = path.name
            frames.append(df)
    if not frames:
        raise FileNotFoundError(f"No metric CSVs found in {ART}")

    df = pd.concat(frames, ignore_index=True, sort=False)
    metric_cols = ["accuracy", "macro_f1", "roc_auc", "pr_auc", "n_samples"]
    for col in metric_cols:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce")
    df = df[df["macro_f1"].notna()].copy()
    df["model_display"] = df["model"].map(compact_name)
    df["model_family"] = df["model"].map(model_family)
    df = df.sort_values("macro_f1", ascending=False).reset_index(drop=True)
    df.insert(0, "rank", np.arange(1, len(df) + 1))
    return df


def save_tables(df: pd.DataFrame) -> pd.DataFrame:
    TABLE_DIR.mkdir(parents=True, exist_ok=True)
    ordered_cols = [
        "rank",
        "model_display",
        "model_family",
        "n_samples",
        "accuracy",
        "macro_f1",
        "roc_auc",
        "pr_auc",
        "source_file",
    ]
    keep = [c for c in ordered_cols if c in df.columns]
    ordered = df[keep].copy()
    ordered.to_csv(TABLE_DIR / "all_model_metrics_ordered.csv", index=False)

    family = (
        df.assign(base_model=df["model_display"].str.replace(r" s\d+$", "", regex=True))
        .groupby(["base_model", "model_family"], as_index=False)
        .agg(
            runs=("macro_f1", "size"),
            accuracy_mean=("accuracy", "mean"),
            accuracy_std=("accuracy", "std"),
            macro_f1_mean=("macro_f1", "mean"),
            macro_f1_std=("macro_f1", "std"),
            roc_auc_mean=("roc_auc", "mean"),
            pr_auc_mean=("pr_auc", "mean"),
        )
        .sort_values("macro_f1_mean", ascending=False)
        .reset_index(drop=True)
    )
    family.insert(0, "rank", np.arange(1, len(family) + 1))
    family.to_csv(TABLE_DIR / "model_family_metrics_summary.csv", index=False)
    return ordered


def plot_horizontal_bar(df: pd.DataFrame) -> Path:
    FIG_DIR.mkdir(parents=True, exist_ok=True)
    plot_df = df.sort_values("macro_f1", ascending=True).copy()
    top3 = set(df.head(3)["model_display"])
    medal_colors = {
        df.iloc[0]["model_display"]: "#d4af37",
        df.iloc[1]["model_display"]: "#b7bcc6",
        df.iloc[2]["model_display"]: "#cd7f32",
    }
    colors = [medal_colors.get(name, "#4c78a8") if name in top3 else "#91a6bf" for name in plot_df["model_display"]]

    height = max(7, 0.42 * len(plot_df) + 1.8)
    fig, ax = plt.subplots(figsize=(12, height))
    bars = ax.barh(plot_df["model_display"], plot_df["macro_f1"], color=colors, edgecolor="white", linewidth=0.8)

    for bar, value in zip(bars, plot_df["macro_f1"]):
        ax.text(value + 0.006, bar.get_y() + bar.get_height() / 2, f"{value:.3f}", va="center", fontsize=9)

    ax.set_xlabel("Macro-F1", fontsize=12, weight="bold")
    ax.set_ylabel("Model", fontsize=12, weight="bold")
    ax.set_title("Final Model Ranking by Macro-F1", fontsize=16, weight="bold", pad=14)
    ax.set_xlim(0, min(1.0, max(0.9, float(plot_df["macro_f1"].max()) + 0.08)))
    ax.grid(axis="x", linestyle="--", alpha=0.25)
    ax.spines[["top", "right", "left"]].set_visible(False)

    note = "Top 3 highlighted. Results use the hard-sample-cleaned normal split from Notebook 9."
    ax.text(0, -0.095, note, transform=ax.transAxes, fontsize=10, color="#555555")
    fig.tight_layout()
    out = FIG_DIR / "final_macro_f1_horizontal_bar_top3.png"
    fig.savefig(out, dpi=220, bbox_inches="tight")
    plt.close(fig)
    return out


def plot_podium(df: pd.DataFrame) -> Path:
    FIG_DIR.mkdir(parents=True, exist_ok=True)
    top3 = df.head(3).copy()
    best_multimodal = df[df["model_family"].eq("multimodal")].head(1)
    best_text = df[df["model_family"].eq("text-only")].head(1)
    multimodal_won = not best_multimodal.empty and not best_text.empty and best_multimodal.iloc[0]["macro_f1"] > best_text.iloc[0]["macro_f1"]

    fig, ax = plt.subplots(figsize=(11, 6.4))
    ax.set_xlim(0, 10)
    ax.set_ylim(0, 7)
    ax.axis("off")

    podium_specs = [
        (3.8, 1.0, 2.4, 4.0, "#d4af37", "1st"),
        (1.2, 1.0, 2.4, 3.0, "#b7bcc6", "2nd"),
        (6.4, 1.0, 2.4, 2.35, "#cd7f32", "3rd"),
    ]
    order = [0, 1, 2]
    for spec, idx in zip(podium_specs, order):
        x, y, w, h, color, label = spec
        row = top3.iloc[idx]
        ax.add_patch(Rectangle((x, y), w, h, facecolor=color, edgecolor="#222222", linewidth=1.2))
        ax.text(x + w / 2, y + h - 0.35, label, ha="center", va="top", fontsize=18, weight="bold", color="#111111")
        wrapped = "\n".join(textwrap.wrap(row["model_display"], width=22))
        ax.text(x + w / 2, y + h / 2 + 0.15, wrapped, ha="center", va="center", fontsize=11, weight="bold")
        ax.text(x + w / 2, y + 0.35, f"Macro-F1 = {row['macro_f1']:.3f}", ha="center", va="bottom", fontsize=11)

    ax.text(5, 6.55, "Did Multimodal Win?", ha="center", fontsize=22, weight="bold")
    answer = "No: the best model is text-only." if not multimodal_won else "Yes: the best model is multimodal."
    ax.text(5, 6.05, answer, ha="center", fontsize=15, weight="bold", color="#9b1c1c" if not multimodal_won else "#1f7a3a")

    if not best_multimodal.empty:
        bm = best_multimodal.iloc[0]
        note = (
            f"Best multimodal: {bm['model_display']} (Macro-F1={bm['macro_f1']:.3f}).\n"
            "Interpretation: images helped in some trials, but strong text encoders still dominated this dataset."
        )
    else:
        note = "No multimodal model found in the metric table."
    ax.text(
        5,
        0.35,
        note,
        ha="center",
        va="bottom",
        fontsize=11,
        color="#333333",
        bbox=dict(boxstyle="round,pad=0.45", facecolor="#fff4d6", edgecolor="#c99a2e"),
    )
    fig.tight_layout()
    out = FIG_DIR / "did_multimodal_win_top3_podium.png"
    fig.savefig(out, dpi=220, bbox_inches="tight")
    plt.close(fig)
    return out


def plot_reproducibility() -> Path:
    FIG_DIR.mkdir(parents=True, exist_ok=True)
    items = [
        ("Saved splits", "data_splits/ + title_only_archive/"),
        ("Saved predictions", "normal_split_*_test_predictions.csv"),
        ("Saved metrics CSVs", "normal_split_*_metrics.csv"),
        ("Fixed seeds", "SEED=42; retrain seeds 42/123/2026"),
        ("Notebook pipeline", "4 merged notebooks + old archive"),
        ("GitHub repository", "README + PROJECT_STRUCTURE + requirements"),
    ]

    fig, ax = plt.subplots(figsize=(11, 6.3))
    ax.set_xlim(0, 10)
    ax.set_ylim(0, 7)
    ax.axis("off")
    ax.text(5, 6.55, "Reproducibility Evidence", ha="center", fontsize=22, weight="bold")
    ax.text(5, 6.15, "What is saved so the experiments can be audited and rerun", ha="center", fontsize=12, color="#555555")

    y_positions = [5.35, 4.55, 3.75, 2.95, 2.15, 1.35]
    for (title, detail), y in zip(items, y_positions):
        ax.text(1.05, y, "✓", fontsize=22, weight="bold", color="#1f7a3a", va="center")
        ax.text(1.55, y + 0.12, title, fontsize=13, weight="bold", va="center")
        ax.text(1.55, y - 0.22, detail, fontsize=10.5, color="#444444", va="center")
        ax.plot([1.0, 9.0], [y - 0.48, y - 0.48], color="#e3e6ea", linewidth=1)

    ax.text(
        5,
        0.35,
        "Large generated data/images/checkpoints are ignored by Git but preserved locally for reruns.",
        ha="center",
        fontsize=10.5,
        color="#333333",
        bbox=dict(boxstyle="round,pad=0.4", facecolor="#eef5ff", edgecolor="#7aa5d8"),
    )
    fig.tight_layout()
    out = FIG_DIR / "reproducibility_checklist_artifact_layout.png"
    fig.savefig(out, dpi=220, bbox_inches="tight")
    plt.close(fig)
    return out


def main() -> None:
    df = load_metrics()
    ordered = save_tables(df)
    outputs = [
        plot_horizontal_bar(df),
        plot_podium(df),
        plot_reproducibility(),
    ]
    print("Saved visuals:")
    for path in outputs:
        print(f"- {path.relative_to(ROOT)}")
    print("\nSaved tables:")
    print(f"- {(TABLE_DIR / 'all_model_metrics_ordered.csv').relative_to(ROOT)}")
    print(f"- {(TABLE_DIR / 'model_family_metrics_summary.csv').relative_to(ROOT)}")
    print("\nAll model results ordered by Macro-F1:")
    print(ordered.to_string(index=False, float_format=lambda x: f"{x:.4f}"))


if __name__ == "__main__":
    main()
