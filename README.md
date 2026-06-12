# Multimodal Fake News Detection

This repository contains a Deep Learning project for fake/real news classification using FakeNewsNet-style text and image data. The work compares text-only baselines, image-only baselines, multimodal fusion models, pretrained image encoders, CLIP alignment diagnostics, rich OCR/BLIP features, voting ensembles, and hard-sample cleaning experiments.

The key scientific finding is that multimodal models do not automatically outperform strong text baselines. Text is the dominant signal in this dataset, while images are useful mainly for controlled ablations, alignment diagnostics, and failure-case analysis.

## GitHub-Ready Layout

```text
notebooks/
  01_Data_Download_EDA_Image_Content_Preparation.ipynb
  02_Cleaning_Leakage_and_Splitting.ipynb
  03_Main_Multimodal_Modeling_All_Experiments.ipynb
  04_Rich_Hard_Sample_and_Final_Retraining_Experiments.ipynb
  old/                            # original notebooks preserved

reports/
  documents/                      # final IEEE report and previous report copy
  figures/                        # all report/EDA figures gathered in one place
  presentation/                   # presentation plan and slide PDF

docs/
  project_brief/                  # assignment and extracted instructions
  notes/                          # original project notes

scripts/
  build_content_title_modeling_splits.py
  generate_final_ieee_report.py
  old_helpers/                    # one-off helper scripts preserved

models/checkpoints/               # local model checkpoints, ignored by Git
artifacts/                        # generated outputs grouped by experiment, ignored by Git
data_splits/                      # local split CSVs, ignored by Git
dataset/, images/                 # local raw data/images, ignored by Git
```

## Recommended Run Order

1. `notebooks/01_Data_Download_EDA_Image_Content_Preparation.ipynb`
   - Checks whether the dataset exists.
   - Performs EDA.
   - Downloads/prepares images.
   - Extracts article content when possible.
   - Builds content/title modeling splits.

2. `notebooks/02_Cleaning_Leakage_and_Splitting.ipynb`
   - Cleans labels/text.
   - Creates valid-image-only splits.
   - Performs ID, title, and image-hash leakage checks.
   - Creates final image-deleaked split files.

3. `notebooks/03_Main_Multimodal_Modeling_All_Experiments.ipynb`
   - Main experiment notebook.
   - Includes text baselines, image-only CNN, CNN+text fusion, ResNet/ConvNeXt fusion, voting, leakage checks, per-source evaluation, and failure-case analysis.

4. `notebooks/04_Rich_Hard_Sample_and_Final_Retraining_Experiments.ipynb`
   - Preserves rich OCR/BLIP/BERT/ResNet50 trials.
   - Preserves GT6 and hard-sample cleaning experiments.
   - Includes final diagnostic retraining with BERT, DistilBERT, RoBERTa, CLIP, Swin, ConvNeXt, and voting.

## Original Title-Only Data

Before article-content fallback was introduced, the original title-only split CSVs were backed up locally in:

```text
data_splits/title_only_archive/
```

This folder is ignored by Git because `data_splits/` can be large, but it is preserved locally so experiments can be compared against the original title-only setup.

## Important Methodology Notes

- Source/domain columns are audit metadata only; they are not used as model features.
- Duplicate normalized titles must not cross train/validation/test boundaries.
- Image-content leakage is checked with SHA-256 image hashes, not only IDs.
- Valid-image-only evaluation is the fairer setting for claims about visual evidence.
- CLIP similarity is an image-text alignment diagnostic, not a factual truth detector.
- Hard-sample removal experiments are diagnostic and should not be presented as the primary fair benchmark because they are informed by prior model errors.

## Key Report Artifacts

- Final report: `reports/documents/Final_Report_IEEE_G11_Multimodal_Fake_News.docx`
- Main figures: `reports/figures/`
- Primary generated tables: `final_artifacts/` locally
- Diagnostic retraining outputs: `normal_split_gt6_plus_trainhard_artifacts/` locally

Large datasets, images, checkpoints, and generated artifacts are intentionally ignored by Git.
