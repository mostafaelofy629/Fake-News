# Multimodal Fake News Detection

This project experiments with FakeNewsNet metadata, downloaded article images, and title text for fake/real news classification.

## Main Workflow
1. Run `01_FakeNewsNet_EDA_FINAL_Enhanced.ipynb` to build cleaned, de-leaked train/validation/test splits and audit label/source/domain shortcuts.
2. Run `02_Image_Download_and_Preparation_MS1_Ready.ipynb` to download images and create multimodal split files.
3. Run `04_Dataset_Credibility_Audit_and_Cleaning.ipynb` to audit image duplicates/leakage and create image-de-leaked valid-image splits.
4. Run `03_Multimodal_Model_Training_FINAL_Enhanced.ipynb` from the final workflow section to train and compare text-only, neural text, optional CLIP alignment, and multimodal models.

## Important Methodology Notes
- Source and domain columns are audit metadata only, not model inputs.
- Duplicate normalized titles must not cross train/validation/test boundaries.
- Placeholder-image multimodal runs are diagnostic because image availability is uneven by source/label.
- Valid-image-only results are the fairer subset for claims about visual evidence.
- Image-content duplicate leakage must be checked, not only ID/title leakage.
- On the current cleaned split, text-only and multimodal fusion are competitive; do not overclaim multimodal superiority without statistical support.
- CLIP alignment is included as an optional baseline because it directly models image-text semantic agreement.

## Key Outputs
- `final_artifacts/FINAL_REPORT_TABLE_model_comparison.csv`
- `final_artifacts/final_title_leakage_audit.csv`
- `final_artifacts/final_image_coverage_by_split_source_label.csv`
- `final_artifacts/final_valid_image_only_model_results.csv`
- `final_artifacts/final_clip_alignment_results.csv`
- `final_artifacts/final_per_source_metrics.csv`
- `final_artifacts/final_failure_cases_categorized.csv`
- `final_artifacts/final_bootstrap_ci_macro_f1.csv`
