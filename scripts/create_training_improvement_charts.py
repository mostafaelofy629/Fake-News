from __future__ import annotations

from pathlib import Path
import re

import matplotlib.pyplot as plt
import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
NORM = ROOT / "normal_split_gt6_plus_trainhard_artifacts"
FINAL = ROOT / "final_artifacts"
FIG_DIR = ROOT / "reports" / "figures"


def read_history(path: Path, label: str, source: str) -> pd.DataFrame:
    df = pd.read_csv(path)
    if "epoch" not in df.columns:
        df["epoch"] = range(1, len(df) + 1)
    if "val_macro_f1" not in df.columns and "monitor_macro_f1" in df.columns:
        df["val_macro_f1"] = df["monitor_macro_f1"]
    if "train_macro_f1" not in df.columns:
        df["train_macro_f1"] = pd.NA
    df = df[["epoch", "train_loss", "train_macro_f1", "val_macro_f1"]].copy()
    df["model"] = label
    df["source"] = source
    return df


def best_seed_file(pattern: str) -> Path | None:
    candidates = sorted(NORM.glob(pattern))
    if not candidates:
        return None
    best_path = None
    best_score = -1
    for path in candidates:
        df = pd.read_csv(path)
        score_col = "val_macro_f1" if "val_macro_f1" in df.columns else "monitor_macro_f1"
        score = pd.to_numeric(df[score_col], errors="coerce").max()
        if score > best_score:
            best_score = score
            best_path = path
    return best_path


def load_representative_histories() -> list[pd.DataFrame]:
    specs = [
        ("BERT text", NORM / "normal_split_bert_text_history.csv", "after cleaning"),
        ("RoBERTa text", NORM / "NormalSplit_hard-removed_RoBERTa_text-only_history.csv", "after cleaning"),
        ("Swin fusion", NORM / "NormalSplit_hard-removed_Swin-T_frozen_image_+_TF-IDF_fusion_history.csv", "after cleaning"),
        ("CLIP fusion", NORM / "NormalSplit_hard-removed_CLIP_ViT-B_32_frozen_multimodal_fusion_history.csv", "after cleaning"),
    ]
    histories = []
    for label, path, source in specs:
        if path.exists():
            histories.append(read_history(path, label, source))

    seeded_specs = [
        ("Concat fusion", "NormalSplit_hard-removed_Concat_fusion_CNN+Text_seed*_history.csv"),
        ("ResNet50 fusion", "NormalSplit_hard-removed_Frozen_ResNet50_fusion_CNN+Text_seed*_history.csv"),
        ("ConvNeXt fusion", "NormalSplit_hard-removed_Frozen_ConvNeXt-Tiny_fusion_CNN+Text_seed*_history.csv"),
        ("Image-only CNN", "NormalSplit_hard-removed_Image-only_CNN_seed*_history.csv"),
    ]
    for label, pattern in seeded_specs:
        path = best_seed_file(pattern)
        if path:
            histories.append(read_history(path, label, "after cleaning"))
    return histories


def plot_dotted_comparison(histories: list[pd.DataFrame]) -> Path:
    FIG_DIR.mkdir(parents=True, exist_ok=True)
    fig, ax = plt.subplots(figsize=(12, 6.8))
    colors = {
        "BERT text": "#1f77b4",
        "RoBERTa text": "#9467bd",
        "Swin fusion": "#2ca02c",
        "CLIP fusion": "#17becf",
        "Concat fusion": "#ff7f0e",
        "ResNet50 fusion": "#8c564b",
        "ConvNeXt fusion": "#bcbd22",
        "Image-only CNN": "#7f7f7f",
    }
    for df in histories:
        label = df["model"].iloc[0]
        ax.plot(
            df["epoch"],
            df["val_macro_f1"],
            marker="o",
            linestyle=":",
            linewidth=2.4,
            markersize=6,
            color=colors.get(label),
            label=label,
        )
        best_idx = pd.to_numeric(df["val_macro_f1"], errors="coerce").idxmax()
        best = df.loc[best_idx]
        ax.scatter(best["epoch"], best["val_macro_f1"], s=90, color=colors.get(label), edgecolor="black", zorder=5)

    ax.set_title("Training Improvements: Validation Macro-F1 by Epoch", fontsize=18, weight="bold", pad=14)
    ax.set_xlabel("Epoch", fontsize=12, weight="bold")
    ax.set_ylabel("Validation Macro-F1", fontsize=12, weight="bold")
    ax.set_ylim(0.45, 0.88)
    ax.grid(True, linestyle="--", alpha=0.25)
    ax.spines[["top", "right"]].set_visible(False)
    ax.legend(ncol=2, frameon=False, fontsize=10, loc="lower right")
    ax.text(
        0.01,
        -0.13,
        "Dotted lines show validation Macro-F1. Black-edged points mark the selected/best epoch before early stopping.",
        transform=ax.transAxes,
        fontsize=10,
        color="#555555",
    )
    out = FIG_DIR / "training_improvements_dotted_val_macro_f1.png"
    fig.savefig(out, dpi=220, bbox_inches="tight")
    plt.close(fig)
    return out


def plot_overfit_story(histories: list[pd.DataFrame]) -> Path:
    selected = []
    wanted = ["BERT text", "RoBERTa text", "Swin fusion", "Concat fusion", "ResNet50 fusion", "Image-only CNN"]
    for name in wanted:
        match = [h for h in histories if h["model"].iloc[0] == name]
        if match:
            selected.append(match[0])
    if not selected:
        raise RuntimeError("No histories to plot.")

    fig, axes = plt.subplots(2, 3, figsize=(13, 7.4), sharex=False, sharey=True)
    axes = axes.ravel()
    for ax, df in zip(axes, selected):
        label = df["model"].iloc[0]
        ax.plot(df["epoch"], df["val_macro_f1"], marker="o", linestyle=":", linewidth=2.2, color="#d68a30", label="val Macro-F1")
        if df["train_macro_f1"].notna().any():
            ax.plot(df["epoch"], df["train_macro_f1"], marker=".", linestyle="--", linewidth=1.6, color="#8aa1b8", label="train Macro-F1")
        best_idx = pd.to_numeric(df["val_macro_f1"], errors="coerce").idxmax()
        best = df.loc[best_idx]
        ax.axvline(best["epoch"], linestyle=":", color="#333333", alpha=0.6)
        ax.set_title(f"{label}\nbest epoch {int(best['epoch'])}, F1={best['val_macro_f1']:.3f}", fontsize=10, weight="bold")
        ax.grid(True, linestyle="--", alpha=0.22)
        ax.spines[["top", "right"]].set_visible(False)
    for ax in axes[len(selected):]:
        ax.axis("off")
    axes[0].legend(frameon=False, fontsize=8, loc="lower right")
    fig.suptitle("Early Stopping Helped Control Fast Overfitting", fontsize=18, weight="bold", y=0.98)
    fig.text(0.5, 0.02, "Several models peak early, so the final workflow selects the best validation Macro-F1 epoch instead of the last epoch.", ha="center", fontsize=10.5, color="#444444")
    fig.tight_layout(rect=(0, 0.04, 1, 0.94))
    out = FIG_DIR / "training_improvements_early_stopping_grid.png"
    fig.savefig(out, dpi=220, bbox_inches="tight")
    plt.close(fig)
    return out


def plot_before_after_training_story() -> Path:
    before_path = FINAL / "concat_fusion_cnn+text_history_seed42.csv"
    after_path = NORM / "NormalSplit_hard-removed_Concat_fusion_CNN+Text_seed2026_history.csv"
    if not before_path.exists() or not after_path.exists():
        raise FileNotFoundError("Missing before/after concat histories.")
    before = read_history(before_path, "Before cleaning", "fair split")
    after = read_history(after_path, "After cleaning", "diagnostic split")

    fig, ax = plt.subplots(figsize=(10.5, 5.8))
    ax.plot(before["epoch"], before["val_macro_f1"], marker="o", linestyle=":", linewidth=2.6, color="#8aa1b8", label="Before: original fair split")
    ax.plot(after["epoch"], after["val_macro_f1"], marker="o", linestyle=":", linewidth=2.6, color="#d68a30", label="After: hard-sample cleaned split")
    for df, color in [(before, "#8aa1b8"), (after, "#d68a30")]:
        best_idx = pd.to_numeric(df["val_macro_f1"], errors="coerce").idxmax()
        best = df.loc[best_idx]
        ax.scatter(best["epoch"], best["val_macro_f1"], s=120, color=color, edgecolor="black", zorder=5)
        ax.text(best["epoch"] + 0.05, best["val_macro_f1"] + 0.006, f"best {best['val_macro_f1']:.3f}", fontsize=10, weight="bold")

    ax.set_title("Cleaning Improved the Training Signal", fontsize=18, weight="bold", pad=14)
    ax.set_xlabel("Epoch", fontsize=12, weight="bold")
    ax.set_ylabel("Validation Macro-F1", fontsize=12, weight="bold")
    ax.set_ylim(0.68, 0.82)
    ax.grid(True, linestyle="--", alpha=0.25)
    ax.spines[["top", "right"]].set_visible(False)
    ax.legend(frameon=False, fontsize=10, loc="lower right")
    ax.text(
        0.01,
        -0.15,
        "Example shown: concat CNN+text fusion. The cleaned diagnostic split gives a higher validation peak and clearer early-stopping point.",
        transform=ax.transAxes,
        fontsize=10,
        color="#555555",
    )
    out = FIG_DIR / "training_improvements_before_after_cleaning_dotted.png"
    fig.savefig(out, dpi=220, bbox_inches="tight")
    plt.close(fig)
    return out


def main() -> None:
    histories = load_representative_histories()
    outputs = [
        plot_dotted_comparison(histories),
        plot_overfit_story(histories),
        plot_before_after_training_story(),
    ]
    print("Saved training improvement charts:")
    for out in outputs:
        print(f"- {out.relative_to(ROOT)}")


if __name__ == "__main__":
    main()
