from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
ART = ROOT / "normal_split_gt6_plus_trainhard_artifacts"
FIG_DIR = ROOT / "reports" / "figures"


MODELS = [
    ("BERT", "normal_split_bert_text_history.csv", "#1f77b4"),
    ("RoBERTa", "NormalSplit_hard-removed_RoBERTa_text-only_history.csv", "#7b3fb2"),
    ("DistilBERT", "NormalSplit_hard-removed_DistilBERT_text-only_history.csv", "#4d8ac8"),
    ("Swin-T + TF-IDF fusion", "NormalSplit_hard-removed_Swin-T_frozen_image_+_TF-IDF_fusion_history.csv", "#2ca02c"),
    ("CLIP ViT-B/32 fusion", "NormalSplit_hard-removed_CLIP_ViT-B_32_frozen_multimodal_fusion_history.csv", "#17becf"),
    ("Concat CNN+Text", "NormalSplit_hard-removed_Concat_fusion_CNN+Text_seed2026_history.csv", "#ff7f0e"),
    ("ResNet50 fusion", "NormalSplit_hard-removed_Frozen_ResNet50_fusion_CNN+Text_seed123_history.csv", "#8c564b"),
    ("Consistency CNN+Text", "NormalSplit_hard-removed_Consistency_fusion_CNN+Text_seed123_history.csv", "#d62728"),
    ("ConvNeXt-Tiny fusion", "NormalSplit_hard-removed_Frozen_ConvNeXt-Tiny_fusion_CNN+Text_seed123_history.csv", "#bcbd22"),
]


def load_history(filename: str) -> pd.DataFrame:
    path = ART / filename
    if not path.exists():
        raise FileNotFoundError(path)
    df = pd.read_csv(path)
    if "epoch" not in df.columns:
        df["epoch"] = range(1, len(df) + 1)
    if "val_macro_f1" not in df.columns and "monitor_macro_f1" in df.columns:
        df["val_macro_f1"] = df["monitor_macro_f1"]
    df["val_macro_f1"] = pd.to_numeric(df["val_macro_f1"], errors="coerce")
    return df[["epoch", "val_macro_f1"]].dropna()


def main() -> None:
    FIG_DIR.mkdir(parents=True, exist_ok=True)

    fig, ax = plt.subplots(figsize=(13.8, 7.4), facecolor="white")

    for label, filename, color in MODELS:
        df = load_history(filename)
        ax.plot(
            df["epoch"],
            df["val_macro_f1"],
            linestyle=":",
            marker="o",
            linewidth=2.6,
            markersize=6.5,
            color=color,
            label=label,
            alpha=0.95,
        )
        best_idx = df["val_macro_f1"].idxmax()
        best = df.loc[best_idx]
        ax.scatter(
            best["epoch"],
            best["val_macro_f1"],
            s=105,
            color=color,
            edgecolor="black",
            linewidth=1.2,
            zorder=5,
        )

    ax.set_title("Training Behavior Across Strong Models", fontsize=22, weight="bold", pad=16)
    ax.set_xlabel("Epoch", fontsize=13, weight="bold")
    ax.set_ylabel("Validation Macro-F1", fontsize=13, weight="bold")
    ax.set_xlim(0.75, 6.25)
    ax.set_ylim(0.74, 0.865)
    ax.grid(True, linestyle="--", alpha=0.25)
    ax.spines[["top", "right"]].set_visible(False)

    ax.legend(
        loc="center left",
        bbox_to_anchor=(1.01, 0.5),
        frameon=False,
        fontsize=10.5,
        title="Model",
        title_fontsize=11,
    )

    ax.text(
        0.01,
        -0.13,
        "Dotted lines show validation Macro-F1 over epochs. Black-edged points mark each model's best validation epoch.",
        transform=ax.transAxes,
        fontsize=11,
        color="#555555",
    )
    ax.text(
        0.01,
        -0.19,
        "TF-IDF and voting are not shown because they do not train over epochs.",
        transform=ax.transAxes,
        fontsize=10.5,
        color="#777777",
    )

    out = FIG_DIR / "slide24_training_behavior_many_models_dotted.png"
    fig.savefig(out, dpi=220, bbox_inches="tight")
    plt.close(fig)
    print(out.relative_to(ROOT))


if __name__ == "__main__":
    main()
