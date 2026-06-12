from __future__ import annotations

from pathlib import Path
from typing import Iterable

import pandas as pd
from docx import Document
from docx.enum.section import WD_SECTION
from docx.enum.text import WD_ALIGN_PARAGRAPH
from docx.oxml import OxmlElement
from docx.oxml.ns import qn
from docx.shared import Inches, Pt


ROOT = Path(".")
FIG = ROOT / "reports" / "figures"
TABLES = ROOT / "reports" / "tables"
ART = ROOT / "final_artifacts"
NORMAL = ROOT / "normal_split_gt6_plus_trainhard_artifacts"
OUT = ROOT / "reports" / "documents" / "Final_Report_IEEE_G11_Multimodal_Fake_News.docx"
GITHUB_URL = "https://github.com/mostafaelofy629/Fake-News.git"


def read_csv(path: str | Path) -> pd.DataFrame:
    path = Path(path)
    if not path.exists():
        return pd.DataFrame()
    return pd.read_csv(path)


split_summary = read_csv(ART / "final_scientific_dataset_split_summary.csv")
content_summary = read_csv(ART / "content_title_modeling_split_summary.csv")
fair_results = read_csv(ART / "report_table_classification_results.csv")
deep_ablation = read_csv(ART / "final_deep_ablation_results_mean_std.csv")
clip_table = read_csv(ART / "report_table_clip_consistency.csv")
failure_summary = read_csv(ART / "final_failure_category_summary.csv")
source_metrics = read_csv(ART / "final_per_source_metrics.csv")
bootstrap_ci = read_csv(ART / "final_bootstrap_ci_macro_f1.csv")
mcnemar = read_csv(ART / "final_mcnemar_tests.csv")
preclean_image_leakage = read_csv(ART / "cleaning_image_hash_leakage_pair_summary.csv")
preclean_image_deleak = read_csv(ART / "cleaning_image_deleakage_audit.csv")
leakage_pair = read_csv(ART / "final_image_content_leakage_pair_summary.csv")
title_deleak = read_csv(ART / "final_deleakage_split_filter_audit.csv")
image_deleak = read_csv(ART / "final_image_deleakage_split_filter_audit.csv")
cleaning_audit = read_csv(NORMAL / "normal_split_cleaning_audit.csv")
normal_results = read_csv(TABLES / "all_model_metrics_ordered.csv")
family_summary = read_csv(TABLES / "model_family_metrics_summary.csv")
manual_audit_metrics = read_csv(TABLES / "manual_validity_audit_before_after_metrics.csv")


def set_font(run, size: float = 9, bold: bool = False, italic: bool = False) -> None:
    run.font.name = "Times New Roman"
    run._element.rPr.rFonts.set(qn("w:eastAsia"), "Times New Roman")
    run.font.size = Pt(size)
    run.bold = bold
    run.italic = italic


def style_document(doc: Document) -> None:
    for section in doc.sections:
        section.top_margin = Inches(0.7)
        section.bottom_margin = Inches(0.7)
        section.left_margin = Inches(0.65)
        section.right_margin = Inches(0.65)

    normal = doc.styles["Normal"]
    normal.font.name = "Times New Roman"
    normal._element.rPr.rFonts.set(qn("w:eastAsia"), "Times New Roman")
    normal.font.size = Pt(9)

    for name in ["Heading 1", "Heading 2", "Heading 3", "List Bullet"]:
        style = doc.styles[name]
        style.font.name = "Times New Roman"
        style._element.rPr.rFonts.set(qn("w:eastAsia"), "Times New Roman")
        style.font.size = Pt(9)
    doc.styles["Heading 1"].font.size = Pt(10)
    doc.styles["Heading 1"].font.bold = True
    doc.styles["Heading 2"].font.size = Pt(9)
    doc.styles["Heading 2"].font.bold = True


def set_columns(section, count: int = 2, space: str = "360") -> None:
    sect_pr = section._sectPr
    cols = sect_pr.xpath("./w:cols")
    cols = cols[0] if cols else OxmlElement("w:cols")
    if not sect_pr.xpath("./w:cols"):
        sect_pr.append(cols)
    cols.set(qn("w:num"), str(count))
    cols.set(qn("w:space"), str(space))


def add_title_block(doc: Document) -> None:
    p = doc.add_paragraph()
    p.alignment = WD_ALIGN_PARAGRAPH.CENTER
    r = p.add_run("Multimodal Fake News Detection with Image-Text Consistency Analysis")
    set_font(r, 18, bold=True)

    for line in [
        "Mohamed Hasan, Mostafa Elofy, Mohamed Ghanem",
        "Group 11 - Deep Learning Final Project",
        f"Repository: {GITHUB_URL}",
    ]:
        p = doc.add_paragraph()
        p.alignment = WD_ALIGN_PARAGRAPH.CENTER
        r = p.add_run(line)
        set_font(r, 10 if "Repository" not in line else 9)


def add_heading(doc: Document, text: str, level: int = 1) -> None:
    p = doc.add_heading(text, level=level)
    p.paragraph_format.space_before = Pt(6)
    p.paragraph_format.space_after = Pt(2)
    for run in p.runs:
        set_font(run, 10 if level == 1 else 9, bold=True)


def add_para(doc: Document, text: str, lead: str | None = None) -> None:
    p = doc.add_paragraph()
    p.paragraph_format.space_after = Pt(3)
    p.paragraph_format.line_spacing = 1.0
    if lead and text.startswith(lead):
        r = p.add_run(lead)
        set_font(r, 9, bold=True)
        r = p.add_run(text[len(lead) :])
        set_font(r, 9)
    else:
        r = p.add_run(text)
        set_font(r, 9)


def add_bullets(doc: Document, items: Iterable[str]) -> None:
    for item in items:
        p = doc.add_paragraph(style="List Bullet")
        p.paragraph_format.space_after = Pt(1)
        r = p.add_run(item)
        set_font(r, 9)


def fmt(value) -> str:
    if pd.isna(value):
        return "-"
    if isinstance(value, float):
        return f"{value:.4f}"
    return str(value)


def add_table(doc: Document, df: pd.DataFrame, caption: str, columns=None, max_rows=None) -> None:
    if df is None or df.empty:
        add_para(doc, f"{caption}: table not available in the current artifact folder.")
        return
    out = df.copy()
    if columns:
        out = out[[c for c in columns if c in out.columns]]
    if max_rows:
        out = out.head(max_rows)

    cap = doc.add_paragraph()
    cap.alignment = WD_ALIGN_PARAGRAPH.CENTER
    r = cap.add_run(caption)
    set_font(r, 8, bold=True)

    table = doc.add_table(rows=1, cols=len(out.columns))
    table.style = "Table Grid"
    for j, col in enumerate(out.columns):
        set_cell(table.rows[0].cells[j], col, bold=True)
    for _, row in out.iterrows():
        cells = table.add_row().cells
        for j, col in enumerate(out.columns):
            set_cell(cells[j], fmt(row[col]))
    doc.add_paragraph()


def set_cell(cell, text, bold: bool = False) -> None:
    cell.text = ""
    p = cell.paragraphs[0]
    p.paragraph_format.space_after = Pt(0)
    r = p.add_run("" if pd.isna(text) else str(text))
    set_font(r, 6.8, bold=bold)


def add_figure(doc: Document, path: str | Path, caption: str, width: float = 3.05) -> None:
    path = Path(path)
    if not path.exists():
        add_para(doc, f"{caption}: figure unavailable at {path}.")
        return
    p = doc.add_paragraph()
    p.alignment = WD_ALIGN_PARAGRAPH.CENTER
    run = p.add_run()
    run.add_picture(str(path), width=Inches(width))
    cap = doc.add_paragraph()
    cap.alignment = WD_ALIGN_PARAGRAPH.CENTER
    r = cap.add_run(caption)
    set_font(r, 8, italic=True)


def add_formula_block(doc: Document) -> None:
    formulas = [
        "Accuracy = (TP + TN) / (TP + TN + FP + FN)",
        "Precision_c = TP_c / (TP_c + FP_c)",
        "Recall_c = TP_c / (TP_c + FN_c)",
        "F1_c = 2 * Precision_c * Recall_c / (Precision_c + Recall_c)",
        "Macro-F1 = (1 / C) * sum_{c=1}^{C} F1_c",
        "CrossEntropy = - sum_{c=1}^{C} y_c log(p_c)",
        "ROC-AUC = integral TPR(FPR) dFPR",
        "PR-AUC = integral Precision(Recall) dRecall",
        "CLIPSim(image, text) = dot(v_img, v_txt) / (||v_img|| ||v_txt||)",
        "InconsistencyScore = 1 - normalized(CLIPSim)",
    ]
    for formula in formulas:
        p = doc.add_paragraph()
        p.alignment = WD_ALIGN_PARAGRAPH.CENTER
        r = p.add_run(formula)
        set_font(r, 8.2, italic=True)


def add_one_column_section(doc: Document) -> None:
    section = doc.add_section(WD_SECTION.CONTINUOUS)
    section.top_margin = Inches(0.7)
    section.bottom_margin = Inches(0.7)
    section.left_margin = Inches(0.65)
    section.right_margin = Inches(0.65)


def add_two_column_section(doc: Document) -> None:
    section = doc.add_section(WD_SECTION.CONTINUOUS)
    section.top_margin = Inches(0.7)
    section.bottom_margin = Inches(0.7)
    section.left_margin = Inches(0.65)
    section.right_margin = Inches(0.65)
    set_columns(section, 2)


def top_source_table() -> pd.DataFrame:
    if source_metrics.empty:
        return pd.DataFrame()
    keep = source_metrics[source_metrics["model"].str.contains("Text-only TF-IDF", na=False)].copy()
    return keep[["source", "n", "accuracy", "macro_f1", "fake_f1", "real_f1"]]


def compact_mcnemar() -> pd.DataFrame:
    if mcnemar.empty:
        return pd.DataFrame()
    keep = mcnemar.copy()
    keep["comparison"] = keep["comparison"].str.replace("Text-only vs ", "", regex=False)
    return keep[["comparison", "n01_A_correct_B_wrong", "n10_A_wrong_B_correct", "p_value"]]


def build_report() -> None:
    OUT.parent.mkdir(parents=True, exist_ok=True)
    doc = Document()
    style_document(doc)
    add_title_block(doc)

    add_heading(doc, "Abstract", 1)
    add_para(
        doc,
        "Abstract-This final report presents a reproducible multimodal fake-news detection project on the FakeNewsNet GossipCop and PolitiFact subset. The project implements a complete image and text pipeline, then compares text-only, image-only, fusion, pretrained visual, pretrained text, CLIP/Swin, voting, and OCR/BLIP enriched trials. The final study shows that the dataset's fake/real label is related to, but not identical with, image-text inconsistency. Therefore, the report separates supervised fake/real classification from image-text alignment diagnostics. On the fair valid-image, image-deleaked benchmark, text-only TF-IDF Logistic Regression reached 0.7576 Macro-F1 and the best CNN/text fusion reached about 0.7640 Macro-F1. After a separate manual validity and hard-sample audit, BERT reached 0.8489 Macro-F1, RoBERTa reached 0.8364, and the best multimodal diagnostic model, Swin-T plus TF-IDF fusion, reached 0.8078. These results show that text remains the strongest signal, while image features are useful mainly for controlled ablations, diagnostics, and failure-case analysis. The main contribution is a careful evaluation workflow with leakage checks, image validity filtering, manual label-audit discussion, failure cases, and reproducible artifacts."
    )
    add_para(
        doc,
        "Index Terms- fake news detection, multimodal learning, FakeNewsNet, image-text consistency, TF-IDF, BERT, RoBERTa, ResNet50, ConvNeXt, Swin Transformer, CLIP, BLIP, OCR."
    )

    add_two_column_section(doc)

    add_heading(doc, "I. INTRODUCTION", 1)
    add_para(
        doc,
        "Fake news on social platforms is rarely only text. It is usually delivered as a package containing a title, an image, a source, and emotional framing. This makes multimodal modeling attractive: a misleading image can strengthen a false article, and a text-only model cannot inspect the visual evidence. The official project aim was to detect inconsistency between image and text in news articles by implementing a simple multimodal model and comparing it with a text-only baseline."
    )
    add_para(
        doc,
        "During the project, an important limitation became central. FakeNewsNet provides fake/real labels for articles, but it does not directly label whether a specific image contradicts a specific title. A fake article can use a relevant celebrity image, and a real article can use a generic or weakly related image. For that reason, this final report treats fake/real classification as the supervised task and image-text consistency as an auxiliary diagnostic."
    )
    add_para(
        doc,
        "The project contributions are: (1) a cleaned image and text data pipeline; (2) text-only, image-only, fusion, transformer, pretrained image, CLIP/Swin, voting, and rich OCR/BLIP trials; (3) leakage checks using IDs, normalized titles, and SHA-256 image hashes; (4) valid-image-only and per-source evaluation; (5) manual validity and hard-sample auditing; and (6) qualitative failure-case analysis. The active repository is included for reproducibility: " + GITHUB_URL,
        lead="The project contributions are:",
    )

    add_heading(doc, "II. RELATED WORK", 1)
    add_para(
        doc,
        "Text-based fake-news detection uses linguistic and topical patterns in headlines or article bodies. Rashkin et al. analyzed language signals in unreliable news and fact-checking corpora [1]. BERT introduced deep contextual text representations [5], while RoBERTa and DistilBERT provide stronger and lighter transformer baselines for text classification. These models motivate our text-only ladder from TF-IDF Logistic Regression to BERT-family classifiers."
    )
    add_para(
        doc,
        "Multimodal fake-news research motivates combining visual and textual evidence. Jin et al. proposed multimodal fusion with co-attention networks for fake-news detection [2]. Khattar et al. proposed MVAE, a multimodal variational autoencoder that jointly models text and image representations [3]. FakeNewsNet provides the public benchmark used in this project [4]. More recent vision-language models such as CLIP [6] and BLIP [7] support image-text similarity and captioning, but they should not be interpreted as factual truth detectors."
    )

    add_heading(doc, "III. DATA AND PREPROCESSING", 1)
    add_para(
        doc,
        "The dataset is a FakeNewsNet-style multimodal subset containing GossipCop and PolitiFact articles. Each row contains an article identifier, title, source, fake/real label, URL metadata, and image information. Labels are normalized to binary fake/real values. Source and domain columns are retained only for audit tables; they are not used as model features because they can create shortcut learning."
    )
    add_table(doc, split_summary, "TABLE I. FINAL VALID-IMAGE IMAGE-DELEAKED SPLIT SUMMARY", ["split", "rows", "fake", "real", "valid_images", "valid_image_coverage_%"])
    add_para(
        doc,
        "The fair multimodal benchmark uses valid-image-only, image-deleaked files with 4,829 training rows, 541 validation rows, and 518 test rows. This avoids giving image models placeholder images and prevents repeated image content from appearing across train, validation, and test."
    )
    add_table(doc, preclean_image_leakage, "TABLE II. IMAGE-CONTENT LEAKAGE FOUND BEFORE DE-LEAKING", ["split_pair", "n_overlapping_image_hashes", "overlap_%_of_smaller_split"])
    add_table(doc, preclean_image_deleak, "TABLE III. ROWS REMOVED TO FIX PRIOR-SPLIT IMAGE OVERLAP", ["split", "rows_before", "rows_after", "rows_removed_for_prior_split_image_overlap"])
    add_para(
        doc,
        "After this de-leaking step, the final SHA-256 image-content audit reported zero train-validation, train-test, and validation-test image-hash overlap. In the report, this final all-clear result is stated in text instead of shown as a zero-filled table."
    )
    add_table(doc, content_summary, "TABLE IV. CONTENT/TITLE MODELING SPLIT FOLDERS", ["section", "split", "rows", "content_available", "content_coverage_%", "title_fallback", "valid_images"], max_rows=6)
    add_para(
        doc,
        "Article-body extraction was attempted, but historical URLs were often unavailable, blocked, or too short. To keep all usable samples, the final text field is text_for_model: article content is used when available and the title is used as fallback. This creates two clean data views: a valid-image multimodal folder and a text-only folder that can contain samples with or without images."
    )
    add_figure(doc, FIG / "plot_08_image_coverage_source_label.png", "Fig. 1. Image coverage by source and label.")
    add_figure(doc, FIG / "manual_validity_audit_before_after_cleaning.png", "Fig. 2. Manual validity audit before and after cleaning.")

    add_heading(doc, "IV. METHODOLOGY", 1)
    add_para(
        doc,
        "The final workflow is model-family based rather than a single model. The text branch includes majority, TF-IDF Logistic Regression, DistilBERT, BERT, and RoBERTa. The image branch includes an image-only CNN, frozen ResNet50, ConvNeXt-Tiny, Swin-T, and CLIP image embeddings. Fusion models concatenate text and image representations; consistency fusion also includes absolute difference and element-wise product features."
    )
    add_para(
        doc,
        "Deep models use cross-entropy loss, Adam or AdamW optimization, lower learning rates, dropout, weight decay, and early stopping based on validation Macro-F1. Frozen pretrained visual encoders reduce overfitting and make training feasible. Voting ensembles combine model probabilities or labels to test whether several weak/medium signals improve robustness."
    )
    add_para(
        doc,
        "The rich multimodal branch uses BERT text embeddings, ResNet50 image embeddings, EasyOCR text extracted from images, and BLIP captions embedded by BERT. This branch was added to make the analysis richer and to test whether explicit visual text and generated captions help. It is interpreted as an exploratory ablation, not as the primary fair benchmark."
    )
    add_formula_block(doc)
    add_figure(doc, FIG / "plot_11_model_architecture.png", "Fig. 3. Overall multimodal fusion architecture.")

    add_heading(doc, "V. EXPERIMENTS AND RESULTS", 1)
    add_para(
        doc,
        "The main fair benchmark is the 518-row valid-image, image-deleaked test split. Macro-F1 is emphasized because it balances fake and real classes. Accuracy, ROC-AUC, PR-AUC, confusion matrices, bootstrap confidence intervals, and McNemar tests are also reported where available."
    )
    add_table(doc, fair_results, "TABLE V. FAIR VALID-IMAGE TEST RESULTS", ["Model", "Input", "Accuracy", "Macro-F1", "ROC-AUC", "PR-AUC"], max_rows=10)
    add_table(doc, deep_ablation, "TABLE VI. DEEP ABLATION MEAN AND STANDARD DEVIATION ACROSS SEEDS", ["model", "accuracy_mean", "accuracy_std", "macro_f1_mean", "macro_f1_std"], max_rows=6)
    add_para(
        doc,
        "The fair benchmark shows a narrow gap. TF-IDF Logistic Regression reached 0.7576 Macro-F1. Concat fusion and consistency fusion reached approximately 0.7635 and 0.7640 Macro-F1. The improvement exists, but it is small; therefore, the project should not claim that multimodal fusion clearly dominates text-only classification on this dataset."
    )
    add_figure(doc, FIG / "plot_16_final_model_comparison.png", "Fig. 4. Fair benchmark model comparison.")
    add_table(doc, bootstrap_ci, "TABLE VII. BOOTSTRAPPED MACRO-F1 CONFIDENCE INTERVALS", ["model", "mean_boot", "ci_low", "ci_high"], max_rows=8)
    add_table(doc, compact_mcnemar(), "TABLE VIII. MCNEMAR TESTS AGAINST TEXT-ONLY BASELINE", ["comparison", "n01_A_correct_B_wrong", "n10_A_wrong_B_correct", "p_value"], max_rows=8)
    add_para(
        doc,
        "The statistical checks support a careful interpretation. Fusion models are competitive, but the McNemar comparisons against TF-IDF are not decisive for the main fusion models. Image-only models are clearly weaker, confirming that images alone are not enough for fake/real classification in this subset."
    )

    add_heading(doc, "VI. MANUAL VALIDITY AUDIT AND DIAGNOSTIC RETRAINING", 1)
    add_para(
        doc,
        "After the fair benchmark, the project performed a stronger diagnostic audit. Nearly 5,888 image-based records were inspected through validity and repeated-error checks. The audit removed 328 invalid-image rows and 519 hard or suspicious rows that remained after image validation, leaving 5,041 rows. This produced a normal shuffle split of 4,035 train, 502 validation, and 504 test records."
    )
    add_table(doc, cleaning_audit, "TABLE IX. MANUAL VALIDITY AND HARD-SAMPLE CLEANING AUDIT", ["stage", "rows"], max_rows=8)
    add_table(doc, manual_audit_metrics, "TABLE X. BEFORE/AFTER CLEANING MACRO-F1 CHANGE", ["model", "before_macro_f1", "after_macro_f1", "delta_pp"], max_rows=8)
    add_para(
        doc,
        "This diagnostic split changed the story. BERT reached 0.8489 Macro-F1, RoBERTa reached 0.8364, and TF-IDF reached 0.8244. The best multimodal diagnostic models were Swin-T plus TF-IDF fusion at 0.8078 and CLIP ViT-B/32 fusion at 0.8057. This suggests that label noise and invalid or misleading image-text pairs were suppressing performance, but also confirms that strong text encoders remain ahead."
    )
    add_table(doc, normal_results, "TABLE XI. DIAGNOSTIC NORMAL-SPLIT RESULTS ORDERED BY MACRO-F1", ["rank", "model_display", "model_family", "accuracy", "macro_f1", "roc_auc", "pr_auc"], max_rows=12)
    add_figure(doc, FIG / "final_macro_f1_horizontal_bar_top3.png", "Fig. 5. Diagnostic model ranking by Macro-F1.")
    add_figure(doc, FIG / "did_multimodal_win_top3_podium.png", "Fig. 6. Top-three podium: did multimodal win?")
    add_figure(doc, FIG / "slide24_training_behavior_many_models_dotted.png", "Fig. 7. Validation Macro-F1 behavior across selected models.")

    add_heading(doc, "VII. IMAGE-TEXT CONSISTENCY ANALYSIS", 1)
    add_para(
        doc,
        "CLIP similarity was evaluated on valid image-title pairs as a proxy for semantic alignment. The key result is negative but important: fake articles were not necessarily less aligned than real articles. In fact, mean CLIP similarity for fake articles was slightly higher than for real articles in the evaluated set."
    )
    add_table(doc, clip_table, "TABLE XII. CLIP IMAGE-TEXT CONSISTENCY ANALYSIS", ["Metric", "Value", "Interpretation"], max_rows=10)
    add_figure(doc, FIG / "clip_similarity_by_class.png", "Fig. 8. CLIP similarity by FakeNewsNet label.")
    add_figure(doc, FIG / "clip_inconsistency_roc_curve.png", "Fig. 9. CLIP inconsistency ROC curve.")
    add_para(
        doc,
        "The conclusion is that CLIP is useful for inspection, but not a standalone fake-news detector here. A fake article can use a semantically matching image while making a false claim, and a real article can use a generic image. This is why the final report separates image-text alignment from factuality classification."
    )
    add_figure(doc, FIG / "fake_high_clip_similarity_grid.png", "Fig. 10. Fake articles with high CLIP similarity.")
    add_figure(doc, FIG / "fake_low_clip_similarity_grid.png", "Fig. 11. Fake articles with low CLIP similarity.")

    add_heading(doc, "VIII. DISCUSSION AND FAILURE ANALYSIS", 1)
    add_para(
        doc,
        "The strongest lesson is that the dataset and protocol matter as much as the model. Text-only models are strong because the label is fake/real, and many fake-news cues appear in wording, topic, and headline style. Image features help in some controlled fusion settings, but they do not automatically solve the task because the visual information is not directly supervised as an inconsistency label."
    )
    add_table(doc, failure_summary, "TABLE XIII. FAILURE-CASE CATEGORY SUMMARY", ["failure_category", "count"])
    add_table(doc, top_source_table(), "TABLE XIV. PER-SOURCE TEXT BASELINE DIAGNOSTIC METRICS", ["source", "n", "accuracy", "macro_f1", "fake_f1", "real_f1"])
    add_para(
        doc,
        "Failure cases often involved short or ambiguous titles, real articles with sensational wording, fake articles with visually relevant images, and source/domain imbalance. The PolitiFact portion is very small in the final valid-image test set, so per-source metrics for it are diagnostic rather than conclusive."
    )
    add_figure(doc, FIG / "plot_14_failure_cases.png", "Fig. 12. Qualitative failure-case examples.")
    add_figure(doc, FIG / "reproducibility_checklist_artifact_layout.png", "Fig. 13. Reproducibility evidence and artifact layout.")

    add_heading(doc, "IX. ENGINEERING CHALLENGES SOLVED", 1)
    add_bullets(
        doc,
        [
            "Image availability bias: missing or invalid images differed by source and label. The final report separates valid-image-only multimodal results from text-only all-row results.",
            "Image-content leakage: exact image reuse was checked with SHA-256 hashes, not only article IDs. The final fair split has zero cross-split image-hash overlap.",
            "Fast overfitting: early neural models peaked quickly. AdamW, lower learning rates, dropout, weight decay, label smoothing, and early stopping were added.",
            "Split alignment: text and multimodal predictions were aligned to the same final test rows before comparison, avoiding misleading cross-split metrics.",
            "Pretrained model friction: BERT, RoBERTa, CLIP, BLIP, EasyOCR, ResNet50, ConvNeXt, and Swin require package and cache management. Optional sections now cache artifacts and skip cleanly when a model is unavailable.",
            "Label-concept mismatch: fake/real labels are not direct image-text inconsistency labels. The final analysis separates classification, CLIP alignment, and manual validity auditing.",
        ],
    )

    add_heading(doc, "X. REPRODUCIBILITY", 1)
    add_para(
        doc,
        "The repository is organized for repeatable execution. Notebook 01 performs dataset checks, EDA, image preparation, and content extraction. Notebook 02 performs cleaning, leakage audits, and split creation. Notebook 03 contains the main model experiments. Notebook 04 preserves rich multimodal, hard-sample, and retraining experiments. Figures are stored under reports/figures, tables under reports/tables, and large local artifacts are intentionally ignored by Git."
    )
    add_para(doc, "The main reproducibility evidence includes saved splits, saved predictions, metrics CSV files, fixed seeds, notebook run order, requirements.txt, README.md, and the GitHub repository link: " + GITHUB_URL)
    add_table(doc, family_summary, "TABLE XV. MODEL FAMILY SUMMARY FOR DIAGNOSTIC RUNS", ["rank", "base_model", "model_family", "runs", "accuracy_mean", "macro_f1_mean", "roc_auc_mean"], max_rows=12)

    add_heading(doc, "XI. TIMELINE AND TEAM CONTRIBUTIONS", 1)
    timeline = pd.DataFrame(
        [
            ["June 1-3, 2026", "EDA, title cleaning, class/source audit", "Mohamed Hasan, Mostafa Elofy"],
            ["June 4-5, 2026", "Image preparation, validation, manifests", "Mohamed Ghanem"],
            ["June 6-7, 2026", "TF-IDF, DistilBERT, BERT/RoBERTa text baselines", "Mostafa Elofy"],
            ["June 8, 2026", "CNN, concat fusion, consistency fusion, ResNet50 fusion", "Mostafa Elofy, Mohamed Ghanem"],
            ["June 9, 2026", "Leakage checks, image-deleaked splits, valid-image evaluation", "Mohamed Hasan, Mostafa Elofy"],
            ["June 10, 2026", "CLIP, Swin, voting, failure cases, per-source evaluation", "All members"],
            ["June 11-12, 2026", "Manual validity audit, final visuals, report, GitHub organization", "All members"],
        ],
        columns=["Date", "Task", "Responsible"],
    )
    add_table(doc, timeline, "TABLE XVI. FINAL TIMELINE WITH STUDENT ASSIGNMENTS")
    add_para(
        doc,
        "Mohamed Hasan contributed to project framing, dataset credibility review, source-bias interpretation, leakage-risk discussion, and final presentation/report logic. He helped ensure that the report did not overclaim image-text inconsistency from fake/real labels."
    )
    add_para(
        doc,
        "Mostafa Elofy implemented the main notebook pipeline, text baselines, transformer trials, fusion models, CLIP/Swin diagnostics, voting, metric tables, report figures, and repository organization. He also maintained reproducibility artifacts and final report generation."
    )
    add_para(
        doc,
        "Mohamed Ghanem focused on image-side preparation and modeling, including image validation, visual branch experiments, ResNet50/ConvNeXt/Swin analysis support, qualitative image inspection, and presentation visuals."
    )

    add_heading(doc, "XII. USE OF GENAI TOOLS", 1)
    add_para(
        doc,
        "Generative AI assistance was used as a coding and writing support tool for notebook organization, debugging, report drafting, and wording refinement. All experiments, model outputs, metrics, tables, and figures reported in this document are based on project artifacts generated in the local repository. GenAI was not used to fabricate results, alter labels, or claim unavailable experiments. The team reviewed and selected the final interpretations and remains responsible for the submitted content."
    )

    add_heading(doc, "XIII. CONCLUSION", 1)
    add_para(
        doc,
        "This project implemented the required image and text pipeline, text-only baseline, multimodal fusion model, and failure-case analysis. The final result is scientifically careful: multimodal models were competitive, but text-only models remained very strong and sometimes stronger, especially after manual validity auditing. CLIP/Swin and rich OCR/BLIP trials add useful analysis, but they do not change the main conclusion. For future work, the most important improvement is not only a larger model; it is better supervision through direct image-text consistency labels, stronger article-body recovery, and more balanced source coverage."
    )

    add_heading(doc, "REFERENCES", 1)
    references = [
        'H. Rashkin, E. Choi, J. Y. Jang, S. Volkova, and Y. Choi, "Truth of Varying Shades: Analyzing Language in Fake News and Political Fact-Checking," in Proc. EMNLP, 2017.',
        'Z. Jin, J. Cao, H. Guo, Y. Zhang, and J. Luo, "Multimodal Fusion with Co-Attention Networks for Fake News Detection," in Proc. ACL, 2017.',
        'S. Khattar, J. S. Goud, M. Gupta, and V. Varma, "MVAE: Multimodal Variational Autoencoder for Fake News Detection," in Proc. WWW Companion, 2019.',
        'K. Shu, D. Mahudeswaran, S. Wang, D. Lee, and H. Liu, "FakeNewsNet: A Data Repository with News Content, Social Context, and Spatiotemporal Information for Studying Fake News on Social Media," Big Data, 2020.',
        'J. Devlin, M.-W. Chang, K. Lee, and K. Toutanova, "BERT: Pre-training of Deep Bidirectional Transformers for Language Understanding," in Proc. NAACL-HLT, 2019.',
        'Y. Liu et al., "RoBERTa: A Robustly Optimized BERT Pretraining Approach," arXiv:1907.11692, 2019.',
        'V. Sanh, L. Debut, J. Chaumond, and T. Wolf, "DistilBERT, a Distilled Version of BERT," arXiv:1910.01108, 2019.',
        'K. He, X. Zhang, S. Ren, and J. Sun, "Deep Residual Learning for Image Recognition," in Proc. CVPR, 2016.',
        'A. Radford et al., "Learning Transferable Visual Models From Natural Language Supervision," in Proc. ICML, 2021.',
        'J. Li et al., "BLIP: Bootstrapping Language-Image Pre-training for Unified Vision-Language Understanding and Generation," in Proc. ICML, 2022.',
        'Z. Liu et al., "Swin Transformer: Hierarchical Vision Transformer using Shifted Windows," in Proc. ICCV, 2021.',
        'Z. Liu et al., "A ConvNet for the 2020s," in Proc. CVPR, 2022.',
        'Y. Zhou and R. Zafarani, "A Survey of Fake News: Fundamental Theories, Detection Methods, and Opportunities," ACM Computing Surveys, 2020.',
    ]
    for i, ref in enumerate(references, 1):
        add_para(doc, f"[{i}] {ref}")

    add_heading(doc, "APPENDIX: MAIN ARTIFACTS", 1)
    add_bullets(
        doc,
        [
            "notebooks/01_Data_Download_EDA_Image_Content_Preparation.ipynb",
            "notebooks/02_Cleaning_Leakage_and_Splitting.ipynb",
            "notebooks/03_Main_Multimodal_Modeling_All_Experiments.ipynb",
            "notebooks/04_Rich_Hard_Sample_and_Final_Retraining_Experiments.ipynb",
            "reports/tables/all_model_metrics_ordered.csv",
            "reports/tables/manual_validity_audit_before_after_metrics.csv",
            "reports/figures/final_macro_f1_horizontal_bar_top3.png",
            "reports/figures/manual_validity_audit_before_after_cleaning.png",
            "reports/figures/reproducibility_checklist_artifact_layout.png",
        ],
    )

    doc.save(OUT)
    print(f"Saved {OUT.resolve()}")


if __name__ == "__main__":
    build_report()
