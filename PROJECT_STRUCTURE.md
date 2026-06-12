# Project Structure and Upload Notes

## What Was Reorganized

- Original notebooks were preserved in `notebooks/legacy/`.
- GitHub-facing merged notebooks were created in `notebooks/`.
- Root-level plots were moved into `reports/figures/`.
- Report and presentation files were moved into `reports/`.
- One-off helper scripts were moved into `scripts/legacy_helpers/`.
- Model checkpoints were moved into `models/checkpoints/`.
- Root-level prediction/history/audit CSVs were moved into `artifacts/`.
- Original title-only split CSVs were backed up locally in `data_splits/legacy_title_only/`.

## Main Notebooks

| Notebook | Purpose |
|---|---|
| `notebooks/01_Data_Download_EDA_Image_Content_Preparation.ipynb` | Dataset checks, EDA, image preparation, article-content extraction, content/title split construction |
| `notebooks/02_Cleaning_Leakage_and_Splitting.ipynb` | Cleaning, credibility audits, valid-image filtering, title/image leakage checks |
| `notebooks/03_Main_Multimodal_Modeling_All_Experiments.ipynb` | Main text, image, multimodal, voting, and failure-analysis experiments |
| `notebooks/04_Rich_Hard_Sample_and_Final_Retraining_Experiments.ipynb` | Rich OCR/BLIP trials, GT6 filtering, hard-sample retraining, BERT/RoBERTa/CLIP/Swin additions |

## Local-Only Large Folders

These folders are intentionally ignored by Git:

- `dataset/`
- `images/`
- `data_splits/`
- `final_artifacts/`
- `normal_split_gt6_plus_trainhard_artifacts/`
- `artifacts/`
- `models/checkpoints/`

Keep them locally for reruns, but do not upload them unless your instructor specifically asks for generated outputs.

## Upload Recommendation

Commit the readable project materials:

- `README.md`
- `PROJECT_STRUCTURE.md`
- `requirements.txt`
- `notebooks/`
- `scripts/`
- `reports/documents/Final_Report_IEEE_G11_Multimodal_Fake_News.docx`
- selected figures in `reports/figures/`

Avoid committing model checkpoints, raw images, extracted datasets, and generated prediction CSVs.
