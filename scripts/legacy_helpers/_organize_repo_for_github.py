from __future__ import annotations

import copy
import shutil
from pathlib import Path

import nbformat
from nbformat.v4 import new_markdown_cell


ROOT = Path(".").resolve()


def ensure_dirs(*paths: str | Path) -> None:
    for path in paths:
        Path(path).mkdir(parents=True, exist_ok=True)


def move_file(src: Path, dst: Path) -> None:
    if not src.exists():
        return
    dst.parent.mkdir(parents=True, exist_ok=True)
    if dst.exists():
        if src.resolve() == dst.resolve():
            return
        stem, suffix = dst.stem, dst.suffix
        counter = 1
        while dst.exists():
            dst = dst.with_name(f"{stem}_{counter}{suffix}")
            counter += 1
    try:
        shutil.move(str(src), str(dst))
    except PermissionError:
        # Word/preview can lock docx files. Preserve the organized copy and leave
        # the locked source in place instead of stopping the repo cleanup.
        shutil.copy2(src, dst)
        print(f"Copied locked file instead of moving: {src} -> {dst}")


def copy_file(src: Path, dst: Path) -> None:
    if not src.exists():
        return
    dst.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy2(src, dst)


def merge_notebooks(output: Path, sources: list[Path], title: str, description: str) -> None:
    merged = nbformat.v4.new_notebook()
    merged.cells.append(new_markdown_cell(f"# {title}\n\n{description}"))
    for src in sources:
        if not src.exists() and (Path("notebooks/legacy") / src.name).exists():
            src = Path("notebooks/legacy") / src.name
        if not src.exists():
            merged.cells.append(new_markdown_cell(f"## Missing Source Notebook\n\n`{src.name}` was not found during reorganization."))
            continue
        nb = nbformat.read(src, as_version=4)
        if "kernelspec" in nb.metadata:
            merged.metadata["kernelspec"] = nb.metadata["kernelspec"]
        if "language_info" in nb.metadata:
            merged.metadata["language_info"] = nb.metadata["language_info"]
        merged.cells.append(
            new_markdown_cell(
                f"---\n\n## Legacy Source: `{src.name}`\n\n"
                "The following cells are preserved from the original notebook during GitHub reorganization."
            )
        )
        merged.cells.extend(copy.deepcopy(nb.cells))
    output.parent.mkdir(parents=True, exist_ok=True)
    nbformat.write(merged, output)


def main() -> None:
    ensure_dirs(
        "notebooks/legacy",
        "reports/figures",
        "reports/documents",
        "reports/presentation",
        "docs/project_brief",
        "docs/notes",
        "scripts/legacy_helpers",
        "models/checkpoints",
        "artifacts/root_predictions",
        "artifacts/root_training_histories",
        "artifacts/root_audits_and_tables",
        "artifacts/ms1",
        "artifacts/experiment_branches",
        "data_splits/legacy_title_only",
    )

    # Preserve the original title-only CSV split before article-content fallback augmentation.
    legacy_split_files = [
        "clean_full.csv",
        "train.csv",
        "val.csv",
        "test.csv",
        "train_multimodal.csv",
        "val_multimodal.csv",
        "test_multimodal.csv",
    ]
    for name in legacy_split_files:
        copy_file(Path("data_splits") / name, Path("data_splits/legacy_title_only") / name)
    Path("data_splits/legacy_title_only/README.md").write_text(
        "# Legacy Title-Only Splits\n\n"
        "These files preserve the pre-content-extraction title-only splits. Later notebooks add "
        "`text_for_model`, which uses article content when available and falls back to title text.\n",
        encoding="utf-8",
    )

    # Create clean GitHub-facing notebook entrypoints while preserving every original notebook.
    merge_notebooks(
        Path("notebooks/01_Data_Download_EDA_Image_Content_Preparation.ipynb"),
        [
            Path("01_FakeNewsNet_EDA_FINAL_Enhanced.ipynb"),
            Path("02_Image_Download_and_Preparation_MS1_Ready.ipynb"),
            Path("06_Article_Content_Extraction_and_Content_Splits.ipynb"),
            Path("07_Build_Content_Title_Modeling_Splits.ipynb"),
        ],
        "Data Download, EDA, Image Preparation, and Content Extraction",
        "Merged entrypoint for dataset availability checks, EDA, image preparation, article-content extraction, and content/title split construction.",
    )
    merge_notebooks(
        Path("notebooks/02_Cleaning_Leakage_and_Splitting.ipynb"),
        [
            Path("02_Dataset_Cleaning_and_Splitting_FINAL.ipynb"),
            Path("04_Dataset_Credibility_Audit_and_Cleaning.ipynb"),
        ],
        "Dataset Cleaning, Credibility Audits, and Leakage-Safe Splitting",
        "Merged entrypoint for cleaning, valid-image filtering, title leakage checks, image-hash leakage checks, and final split creation.",
    )
    merge_notebooks(
        Path("notebooks/03_Main_Multimodal_Modeling_All_Experiments.ipynb"),
        [Path("03_Multimodal_Model_Training_FINAL_Enhanced.ipynb")],
        "Main Multimodal Modeling and Evaluation Experiments",
        "Primary modeling notebook containing text baselines, image-only, fusion, voting, failure analysis, and final diagnostics.",
    )
    merge_notebooks(
        Path("notebooks/04_Rich_Hard_Sample_and_Final_Retraining_Experiments.ipynb"),
        [
            Path("05_Run_Rich_Multimodal_Branch.ipynb"),
            Path("08_Filtered_GT6_Retrain_Experiment.ipynb"),
            Path("09_GT6_Clean_All_CrossValidation_Preparation.ipynb"),
        ],
        "Rich Multimodal, Hard-Sample, and Final Retraining Experiments",
        "Preserves the rich OCR/BLIP branch, GT6 filtering experiment, and final hard-sample-cleaned normal-split retraining notebook.",
    )

    for nb in ROOT.glob("*.ipynb"):
        move_file(nb, Path("notebooks/legacy") / nb.name)

    # Figures: root-level plots are moved; final-artifact report figures are copied for convenience.
    for png in ROOT.glob("plot*.png"):
        move_file(png, Path("reports/figures") / png.name)
    for png in Path("final_artifacts").glob("*.png") if Path("final_artifacts").exists() else []:
        copy_file(png, Path("reports/figures") / png.name)

    # Documents and presentation materials.
    for name in [
        "Final_Report_IEEE_G11_Multimodal_Fake_News.docx",
        "Final_Report_IEEE_G11_Multimodal_Fake_News.previous.docx",
        "report.docx",
        "report_extracted.txt",
    ]:
        move_file(Path(name), Path("reports/documents") / name)
    for name in ["PRESENTATION_PLAN_AND_SCRIPT.md", "Yellow and Black Modern True Crime Presentation.pdf"]:
        move_file(Path(name), Path("reports/presentation") / name)
    for name in ["project-instructions.pdf", "project_instructions_extracted.txt"]:
        move_file(Path(name), Path("docs/project_brief") / name)
    for name in ["MUST READ.txt"]:
        move_file(Path(name), Path("docs/notes") / name)

    # Scripts and helper scripts.
    for name in ["build_content_title_modeling_splits.py", "generate_final_ieee_report.py"]:
        move_file(Path(name), Path("scripts") / name)
    for py in ROOT.glob("_*.py"):
        move_file(py, Path("scripts/legacy_helpers") / py.name)

    # Checkpoints and generated root-level experiment outputs.
    for pt in ROOT.glob("*.pt"):
        move_file(pt, Path("models/checkpoints") / pt.name)
    for csv in ROOT.glob("predictions_*.csv"):
        move_file(csv, Path("artifacts/root_predictions") / csv.name)
    for csv in ROOT.glob("history_*.csv"):
        move_file(csv, Path("artifacts/root_training_histories") / csv.name)
    for csv in ROOT.glob("ms1_*.csv"):
        move_file(csv, Path("artifacts/ms1") / csv.name)
    for csv in ROOT.glob("final_*.csv"):
        move_file(csv, Path("artifacts/root_audits_and_tables") / csv.name)
    for csv in ROOT.glob("duplicate_*.csv"):
        move_file(csv, Path("artifacts/root_audits_and_tables") / csv.name)

    # Older branch artifact folders are grouped under ignored artifacts/.
    for folder in [
        "filtered_gt6_retrain_artifacts",
        "gt6_cleaned_cv_artifacts",
        "gt6_cleaned_cv_retrain_artifacts",
        "image_manifests",
    ]:
        move_file(Path(folder), Path("artifacts/experiment_branches") / folder)

    print("Repository organization complete.")


if __name__ == "__main__":
    main()
