from pathlib import Path

import numpy as np
import pandas as pd


DATA_SPLIT_DIR = Path("data_splits")
FINAL_ARTIFACT_DIR = Path("final_artifacts")

MULTIMODAL_OUT_DIR = DATA_SPLIT_DIR / "content_title_valid_image_multimodal"
TEXT_ONLY_OUT_DIR = DATA_SPLIT_DIR / "content_title_text_only_all"

MULTIMODAL_OUT_DIR.mkdir(parents=True, exist_ok=True)
TEXT_ONLY_OUT_DIR.mkdir(parents=True, exist_ok=True)
FINAL_ARTIFACT_DIR.mkdir(exist_ok=True)

CONTENT_CACHE_PATH = FINAL_ARTIFACT_DIR / "article_content_cache.csv"
MIN_CONTENT_CHARS = 250


def cache_key(url: str) -> str:
    import hashlib

    if not isinstance(url, str) or not url.strip():
        return ""
    url = url.strip()
    if not url.lower().startswith(("http://", "https://")):
        url = "https://" + url
    return hashlib.sha256(url.encode("utf-8")).hexdigest()


def load_cache() -> pd.DataFrame:
    if CONTENT_CACHE_PATH.exists():
        cache = pd.read_csv(CONTENT_CACHE_PATH)
        if "url_key" in cache.columns and "content" in cache.columns:
            return cache.drop_duplicates("url_key", keep="last")
    return pd.DataFrame(columns=["url_key", "content", "status", "content_len_chars", "normalized_url"])


def first_existing(candidates):
    for path in candidates:
        if path.exists():
            return path
    raise FileNotFoundError(f"No existing file found. Checked: {[str(p) for p in candidates]}")


def source_path_for(split: str, mode: str) -> Path:
    if mode == "multimodal":
        return first_existing(
            [
                DATA_SPLIT_DIR / f"{split}_valid_images_only_image_deleaked_with_content.csv",
                DATA_SPLIT_DIR / f"{split}_valid_images_only_image_deleaked.csv",
                DATA_SPLIT_DIR / f"{split}_valid_images_only_with_content.csv",
                DATA_SPLIT_DIR / f"{split}_valid_images_only.csv",
                DATA_SPLIT_DIR / f"{split}_multimodal.csv",
            ]
        )
    if mode == "text_only":
        return first_existing(
            [
                DATA_SPLIT_DIR / f"{split}_with_content.csv",
                DATA_SPLIT_DIR / f"{split}.csv",
            ]
        )
    raise ValueError(mode)


def attach_content(df: pd.DataFrame, cache: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    if "content" not in out.columns:
        out["content"] = ""
    if "content_status" not in out.columns:
        out["content_status"] = "not_attempted"
    if "content_len_chars" not in out.columns:
        out["content_len_chars"] = out["content"].fillna("").astype(str).str.len()

    if "news_url" in out.columns and len(cache):
        cache_cols = [c for c in ["url_key", "content", "status", "content_len_chars", "normalized_url"] if c in cache.columns]
        cache_small = cache[cache_cols].copy()
        cache_small = cache_small.rename(columns={"status": "cache_content_status", "content": "cache_content"})
        out["url_key"] = out["news_url"].map(cache_key)
        out = out.merge(cache_small, on="url_key", how="left")
        cache_content = out.get("cache_content", pd.Series("", index=out.index)).fillna("").astype(str)
        current_content = out["content"].fillna("").astype(str)
        use_cache = cache_content.str.len() > current_content.str.len()
        out["content"] = np.where(use_cache, cache_content, current_content)
        out["content_status"] = np.where(
            use_cache,
            out.get("cache_content_status", pd.Series("cache", index=out.index)).fillna("cache"),
            out["content_status"].fillna("not_attempted"),
        )
        if "content_len_chars_y" in out.columns:
            out = out.drop(columns=["content_len_chars_y"])
        if "content_len_chars_x" in out.columns:
            out = out.rename(columns={"content_len_chars_x": "content_len_chars"})
        out = out.drop(columns=["cache_content", "cache_content_status"], errors="ignore")

    out["content"] = out["content"].fillna("").astype(str)
    out["content_len_chars"] = out["content"].str.len()
    out["content_len_words"] = out["content"].str.split().map(len)
    out["content_available"] = out["content_len_chars"].ge(MIN_CONTENT_CHARS)
    out["title"] = out["title"].fillna("").astype(str) if "title" in out.columns else ""
    out["text_for_model"] = np.where(out["content_available"], out["content"], out["title"])
    out["text_source_for_model"] = np.where(out["content_available"], "content", "title_fallback")
    return out


def ensure_valid_image_only(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    if "valid_image_path" in out.columns:
        out = out[out["valid_image_path"].astype(bool)].copy()
    elif "image_path_primary" in out.columns:
        out = out[out["image_path_primary"].apply(lambda p: isinstance(p, str) and Path(p).exists())].copy()
        out["valid_image_path"] = True
    return out.reset_index(drop=True)


def ordered_columns(df: pd.DataFrame, mode: str):
    lead = [
        "id",
        "label",
        "label_id",
        "source",
        "title",
        "content",
        "text_for_model",
        "text_source_for_model",
        "content_available",
        "content_status",
        "content_len_chars",
        "content_len_words",
        "news_url",
    ]
    if mode == "multimodal":
        lead += ["valid_image_path", "image_path_primary", "image_paths", "image_sha256"]
    return [c for c in lead if c in df.columns] + [c for c in df.columns if c not in lead]


def build():
    cache = load_cache()
    summary_rows = []
    for split in ["train", "val", "test"]:
        for mode, out_dir in [("multimodal", MULTIMODAL_OUT_DIR), ("text_only", TEXT_ONLY_OUT_DIR)]:
            src = source_path_for(split, mode)
            df = pd.read_csv(src)
            df["split"] = split
            if mode == "multimodal":
                df = ensure_valid_image_only(df)
            df = attach_content(df, cache)
            df = df[ordered_columns(df, mode)]
            out_path = out_dir / f"{split}.csv"
            df.to_csv(out_path, index=False)
            summary_rows.append(
                {
                    "section": mode,
                    "split": split,
                    "source_file": str(src),
                    "output_file": str(out_path),
                    "rows": len(df),
                    "content_available": int(df["content_available"].sum()),
                    "content_coverage_%": round(100 * df["content_available"].mean(), 2) if len(df) else 0,
                    "title_fallback": int((df["text_source_for_model"] == "title_fallback").sum()),
                    "valid_images": int(df["valid_image_path"].astype(bool).sum()) if "valid_image_path" in df.columns else np.nan,
                }
            )

    summary = pd.DataFrame(summary_rows)
    summary_path = FINAL_ARTIFACT_DIR / "content_title_modeling_split_summary.csv"
    summary.to_csv(summary_path, index=False)

    for out_dir, description in [
        (MULTIMODAL_OUT_DIR, "Rows with valid images for multimodal models. Uses content when available, otherwise title."),
        (TEXT_ONLY_OUT_DIR, "All split rows for text-only models, regardless of image availability. Uses content when available, otherwise title."),
    ]:
        readme = out_dir / "README.md"
        readme.write_text(
            f"# {out_dir.name}\n\n{description}\n\n"
            "Main modeling columns:\n\n"
            "- `title`\n"
            "- `content`\n"
            "- `text_for_model`\n"
            "- `text_source_for_model`\n"
            "- `label` / `label_id`\n\n"
            "These files preserve split membership and do not create a new random split.\n",
            encoding="utf-8",
        )
    return summary


if __name__ == "__main__":
    result = build()
    print(result.to_string(index=False))
