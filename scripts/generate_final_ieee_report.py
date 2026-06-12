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
ART = ROOT / "final_artifacts"
OUT = ROOT / "Final_Report_IEEE_G11_Multimodal_Fake_News.docx"
GITHUB_URL = "https://github.com/mostafaelofy629/Fake-News.git"


def read_csv(path: str | Path) -> pd.DataFrame:
    path = Path(path)
    return pd.read_csv(path) if path.exists() else pd.DataFrame()


classification = read_csv(ART / "report_table_classification_results.csv")
clip_table = read_csv(ART / "report_table_clip_consistency.csv")
dataset_summary = read_csv(ART / "final_scientific_dataset_split_summary.csv")
deep = read_csv(ART / "final_deep_ablation_results_mean_std.csv")
content_splits = read_csv(ART / "content_title_modeling_split_summary.csv")
failure_summary = read_csv(ART / "final_failure_category_summary.csv")


def set_cell_text(cell, text, size=7.5, bold=False):
    cell.text = ""
    p = cell.paragraphs[0]
    run = p.add_run("" if pd.isna(text) else str(text))
    run.font.name = "Times New Roman"
    run._element.rPr.rFonts.set(qn("w:eastAsia"), "Times New Roman")
    run.font.size = Pt(size)
    run.bold = bold


def set_margins(section, top=0.7, bottom=0.7, left=0.65, right=0.65):
    section.top_margin = Inches(top)
    section.bottom_margin = Inches(bottom)
    section.left_margin = Inches(left)
    section.right_margin = Inches(right)


def set_columns(section, count=2, space="360"):
    sect_pr = section._sectPr
    cols = sect_pr.xpath("./w:cols")
    cols = cols[0] if cols else OxmlElement("w:cols")
    if not sect_pr.xpath("./w:cols"):
        sect_pr.append(cols)
    cols.set(qn("w:num"), str(count))
    cols.set(qn("w:space"), str(space))


def style_doc(doc: Document):
    styles = doc.styles
    normal = styles["Normal"]
    normal.font.name = "Times New Roman"
    normal._element.rPr.rFonts.set(qn("w:eastAsia"), "Times New Roman")
    normal.font.size = Pt(9)

    for name in ["Heading 1", "Heading 2", "Heading 3"]:
        st = styles[name]
        st.font.name = "Times New Roman"
        st._element.rPr.rFonts.set(qn("w:eastAsia"), "Times New Roman")
        st.font.bold = True
    styles["Heading 1"].font.size = Pt(10)
    styles["Heading 2"].font.size = Pt(9)
    styles["Heading 3"].font.size = Pt(9)


def add_title(doc: Document):
    p = doc.add_paragraph()
    p.alignment = WD_ALIGN_PARAGRAPH.CENTER
    r = p.add_run("Multimodal Fake News Detection with Image-Text Consistency Analysis")
    r.bold = True
    r.font.name = "Times New Roman"
    r.font.size = Pt(18)

    p = doc.add_paragraph()
    p.alignment = WD_ALIGN_PARAGRAPH.CENTER
    r = p.add_run("Mohamed Hasan, Mostafa Elofy, Mohamed Ghanem")
    r.font.name = "Times New Roman"
    r.font.size = Pt(10)

    p = doc.add_paragraph()
    p.alignment = WD_ALIGN_PARAGRAPH.CENTER
    r = p.add_run("Group 11 - Deep Learning Final Project")
    r.font.name = "Times New Roman"
    r.font.size = Pt(10)

    p = doc.add_paragraph()
    p.alignment = WD_ALIGN_PARAGRAPH.CENTER
    r = p.add_run(f"Repository: {GITHUB_URL}")
    r.font.name = "Times New Roman"
    r.font.size = Pt(9)


def add_heading(doc: Document, text: str, level=1):
    prefixes = {
        1: text,
        2: text,
        3: text,
    }
    h = doc.add_heading(prefixes[level], level=level)
    h.paragraph_format.space_before = Pt(6)
    h.paragraph_format.space_after = Pt(2)
    return h


def add_para(doc: Document, text: str, bold_lead: str | None = None):
    p = doc.add_paragraph()
    p.paragraph_format.space_after = Pt(3)
    p.paragraph_format.line_spacing = 1.0
    if bold_lead and text.startswith(bold_lead):
        r = p.add_run(bold_lead)
        r.bold = True
        r.font.name = "Times New Roman"
        r.font.size = Pt(9)
        rest = text[len(bold_lead) :]
        r = p.add_run(rest)
    else:
        r = p.add_run(text)
    r.font.name = "Times New Roman"
    r._element.rPr.rFonts.set(qn("w:eastAsia"), "Times New Roman")
    r.font.size = Pt(9)
    return p


def add_bullets(doc: Document, items: Iterable[str]):
    for item in items:
        p = doc.add_paragraph(style="List Bullet")
        p.paragraph_format.space_after = Pt(1)
        r = p.add_run(item)
        r.font.name = "Times New Roman"
        r.font.size = Pt(9)


def fmt(x):
    if pd.isna(x):
        return "-"
    if isinstance(x, float):
        return f"{x:.4f}"
    return str(x)


def add_df_table(doc: Document, df: pd.DataFrame, caption: str, columns=None, max_rows=None):
    if df is None or df.empty:
        add_para(doc, f"{caption}: table unavailable.")
        return
    if columns:
        df = df[[c for c in columns if c in df.columns]].copy()
    if max_rows:
        df = df.head(max_rows)
    cap = doc.add_paragraph()
    cap.alignment = WD_ALIGN_PARAGRAPH.CENTER
    r = cap.add_run(caption)
    r.bold = True
    r.font.name = "Times New Roman"
    r.font.size = Pt(8)
    table = doc.add_table(rows=1, cols=len(df.columns))
    table.style = "Table Grid"
    for j, col in enumerate(df.columns):
        set_cell_text(table.rows[0].cells[j], col, size=7, bold=True)
    for _, row in df.iterrows():
        cells = table.add_row().cells
        for j, col in enumerate(df.columns):
            set_cell_text(cells[j], fmt(row[col]), size=6.7)
    doc.add_paragraph()


def add_figure(doc: Document, path: str | Path, caption: str, width=3.05):
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
    r.font.name = "Times New Roman"
    r.font.size = Pt(8)
    r.italic = True


def add_formula_block(doc: Document):
    formulas = [
        "Accuracy = (TP + TN) / (TP + TN + FP + FN)",
        "Precision_c = TP_c / (TP_c + FP_c)",
        "Recall_c = TP_c / (TP_c + FN_c)",
        "F1_c = 2 * Precision_c * Recall_c / (Precision_c + Recall_c)",
        "Macro-F1 = (1 / C) * sum_{c=1}^{C} F1_c",
        "CrossEntropy = - sum_{c=1}^{C} y_c log(p_c)",
        "CLIPSim(x_img, x_txt) = (v_img · v_txt) / (||v_img|| ||v_txt||)",
        "CLIPSimNorm = (CLIPSim - min(CLIPSim)) / (max(CLIPSim) - min(CLIPSim))",
        "InconsistencyScore = 1 - CLIPSimNorm",
        "Cohen_d = (mean_real - mean_fake) / pooled_std",
    ]
    for f in formulas:
        p = doc.add_paragraph()
        p.alignment = WD_ALIGN_PARAGRAPH.CENTER
        r = p.add_run(f)
        r.font.name = "Times New Roman"
        r.font.size = Pt(8.5)
        r.italic = True


def main():
    doc = Document()
    style_doc(doc)
    set_margins(doc.sections[0], top=0.7, bottom=0.7, left=0.65, right=0.65)

    add_title(doc)

    add_heading(doc, "Abstract", 1)
    add_para(
        doc,
        "This report presents the final implementation and evaluation of a multimodal fake-news detection project using the FakeNewsNet GossipCop and PolitiFact subset. The project began with a text-only TF-IDF baseline and was extended into a controlled multimodal evaluation platform with image-only, concat-fusion, consistency-fusion, frozen ResNet50, transformer text, CLIP alignment, late-fusion, and OCR/BLIP/BERT/ResNet50 enriched trials. Because FakeNewsNet provides fake/real labels rather than ground-truth image-text consistency labels, CLIP is used as a proxy alignment metric rather than as a direct fake-news detector. The strongest results show that title text remains the dominant signal: TF-IDF Logistic Regression reached 0.7576 macro-F1 on the final image-deleaked valid-image test split, while the best consistency-fusion model reached 0.7608 macro-F1 and late fusion reached 0.7602. Image-only and CLIP-only baselines were substantially weaker. The final system therefore contributes not only model comparisons but also leakage checks, valid-image-only evaluation, per-source analysis, failure cases, and a scientifically honest discussion of why visual consistency does not imply factual truth.",
    )
    add_para(
        doc,
        "Index Terms- fake news detection, multimodal learning, FakeNewsNet, CLIP, ResNet50, BERT, OCR, BLIP, image-text consistency, late fusion.",
    )

    # Two column body from here.
    sec = doc.add_section(WD_SECTION.CONTINUOUS)
    set_margins(sec, top=0.7, bottom=0.7, left=0.65, right=0.65)
    set_columns(sec, 2)

    add_heading(doc, "I. Introduction", 1)
    add_para(
        doc,
        "Online misinformation is often multimodal: a short claim, a familiar image, and a source page combine to create an impression of credibility. This project studies fake/real news classification under that multimodal setting. The original project description asked for a simple CNN/text fusion system and qualitative failure-case analysis. During implementation, an important scientific distinction emerged: FakeNewsNet labels articles as fake or real, but it does not directly label whether a specific image contradicts a specific headline. Therefore, this final report separates fake-news factuality classification from image-text consistency analysis."
    )
    add_para(
        doc,
        "The final objective is to classify fake versus real articles while using CLIP similarity as a proxy diagnostic for image-title semantic alignment. A fake article may use an image that correctly matches a named celebrity or event while still making a false textual claim; in that case, image-text similarity can be high. Conversely, a real article may have a generic or weakly related image. This distinction guides the interpretation of every multimodal result in the report."
    )
    add_para(
        doc,
        "Contributions: (1) a reproducible FakeNewsNet preprocessing and image-validation pipeline; (2) multiple final baselines, including majority, TF-IDF Logistic Regression, transformer text, image-only CNN, concat fusion, consistency fusion, frozen ResNet50 fusion, late fusion, CLIP score-only analysis, and OCR/BLIP/BERT/ResNet50 enriched fusion; (3) leakage checks for image reuse, title overlap, and source shortcuts; (4) valid-image-only and per-source evaluation; and (5) qualitative failure-case groups that show why CLIP alone is insufficient for fake-news detection.",
        bold_lead="Contributions:",
    )

    add_heading(doc, "II. Related Work", 1)
    add_para(
        doc,
        "Text-only fake-news detection remains a strong baseline because headlines and article titles contain linguistic, topical, and stylistic signals. Rashkin et al. studied language patterns in unreliable news and fact-checking corpora, motivating lexical and semantic baselines. BERT introduced contextual language modeling and provides a stronger pretrained encoder for text classification. In this project, TF-IDF Logistic Regression and DistilBERT/BERT-style classifiers serve as text baselines."
    )
    add_para(
        doc,
        "Multimodal fake-news detection has been studied through attention, fusion, and generative representations. Jin et al. proposed multimodal fusion with co-attention for fake-news detection, explicitly motivating joint reasoning over visual and textual cues. Khattar et al. proposed MVAE, which learns a shared multimodal latent representation. FakeNewsNet, introduced by Shu et al., provides a benchmark with news content, social context, and multimedia metadata. More recent vision-language tools such as CLIP and BLIP allow image-text alignment and caption generation, but they should not be interpreted as external fact verification systems."
    )

    add_heading(doc, "III. Data and Preprocessing", 1)
    add_para(
        doc,
        "The dataset is the FakeNewsNet multimodal subset, mainly GossipCop and PolitiFact. Each row includes an article identifier, title, URL, source, label, tweet metadata, and image-related information. Labels are standardized as real=0 and fake=1. Source and domain fields are used only for auditing and are deliberately excluded from model inputs to reduce shortcut learning."
    )
    add_df_table(doc, dataset_summary, "TABLE I. FINAL IMAGE-DELEAKED VALID-IMAGE SPLIT SUMMARY", ["split", "rows", "fake", "real", "valid_images", "valid_image_coverage_%"])
    add_para(
        doc,
        "The final headline multimodal experiments use valid-image-only image-deleaked splits: 4829 training rows, 541 validation rows, and 518 test rows. This choice avoids comparing image-based models on placeholder or missing images. A separate text-only folder was also built for all rows regardless of image availability, and a content-title folder was created to support future article-body experiments. Current article-body extraction coverage is low because many historical URLs are dead, blocked, paywalled, or redirected; therefore title fallback is preserved."
    )
    if not content_splits.empty:
        add_df_table(doc, content_splits, "TABLE II. CONTENT/TITLE MODELING SPLIT FOLDERS", ["section", "split", "rows", "content_available", "content_coverage_%", "title_fallback", "valid_images"], max_rows=6)
    add_para(
        doc,
        "Preprocessing includes title cleaning, label normalization, image path validation, valid-image flags, SHA-256 image hashing for leakage checks, image-deleaked split filtering, TF-IDF vectorization, PyTorch image transforms, CLIP score generation, and cached rich features from OCR, BLIP, BERT, and ResNet50. The repository link included for reproducibility is: " + GITHUB_URL
    )
    add_figure(doc, "plot_15_image_coverage_by_label.png", "Fig. 1. Image coverage by label after image preparation.")

    add_heading(doc, "IV. Methodology", 1)
    add_para(
        doc,
        "The final workflow intentionally evaluates several families of models. Text-only baselines include a majority classifier, TF-IDF Logistic Regression, and a transformer text classifier. Image-only CNN measures whether visual features alone contain useful fake/real signal. Multimodal fusion models combine text and image representations by concatenation or by consistency-oriented features such as absolute difference and element-wise product. A frozen ResNet50 fusion model tests whether a stronger pretrained image encoder improves performance."
    )
    add_para(
        doc,
        "Late fusion combines calibrated scores rather than raw high-dimensional embeddings. It uses text fake probability, CLIP similarity, CLIP inconsistency score, and the valid-image flag; threshold selection is performed only on the validation split. The rich trial uses frozen BERT text embeddings, frozen ResNet50 image embeddings, EasyOCR text embedded by BERT, and BLIP captions embedded by BERT, followed by a small MLP. This branch was added as an advanced ablation and report enrichment rather than as the assumed final model."
    )
    add_para(
        doc,
        "CLIP-based evaluation encodes the image and title into a shared embedding space and computes cosine similarity. It is used to analyze image-title alignment, not to verify factual truth. This avoids the conceptual mistake of treating FakeNewsNet fake/real labels as image-text consistency labels."
    )
    add_formula_block(doc)
    add_figure(doc, "plot_11_model_architecture.png", "Fig. 2. Project model architecture and fusion concept.")

    add_heading(doc, "V. Experiments and Results", 1)
    add_para(
        doc,
        "The final experiments use accuracy, macro-F1, macro precision, macro recall, ROC-AUC, and PR-AUC where probability scores are available. Macro-F1 is emphasized because it treats real and fake classes equally. The final report table includes all major trials conducted during the project."
    )
    add_df_table(
        doc,
        classification,
        "TABLE III. FINAL FAKE/REAL CLASSIFICATION RESULTS",
        ["Model", "Input", "Accuracy", "Macro-F1", "Precision Macro", "Recall Macro", "ROC-AUC", "PR-AUC"],
        max_rows=12,
    )
    add_figure(doc, "plot_16_final_model_comparison.png", "Fig. 3. Final model comparison across text, image-only, and multimodal trials.")
    add_para(
        doc,
        "The best models are clustered around 0.76 macro-F1. TF-IDF Logistic Regression reaches 0.7576 macro-F1, consistency fusion reaches 0.7608, and late fusion reaches 0.7602. The gain from multimodal fusion is therefore small, not dramatic. This is a meaningful finding: in this dataset, title text is the dominant supervised signal, while images contribute limited additional information for fake/real labels."
    )
    add_para(
        doc,
        "Image-only CNN reaches only 0.5544 macro-F1, showing that visual content alone is weak. Frozen ResNet50 fusion is competitive but lower than the simpler consistency fusion. The rich OCR/BLIP/BERT/ResNet50 MLP is valuable as a trial but underperforms the simpler baselines, likely because OCR and generated captions introduce noisy features and the training set is too small for a high-dimensional frozen-feature MLP."
    )
    add_figure(doc, ART / "late_fusion_confusion_matrix.png", "Fig. 4. Late-fusion confusion matrix on the final test split.")
    add_figure(doc, ART / "rich_multimodal_confusion_matrix.png", "Fig. 5. Rich OCR/BLIP/BERT/ResNet50 trial confusion matrix.")

    add_heading(doc, "VI. Image-Text Consistency Analysis", 1)
    add_para(
        doc,
        "CLIP analysis was performed on valid image-title pairs. The mean CLIP similarity for fake articles is slightly higher than for real articles, and the inconsistency score obtains ROC-AUC below random. This confirms that CLIP similarity alone is not a fake-news detector for this dataset."
    )
    add_df_table(doc, clip_table, "TABLE IV. CLIP IMAGE-TEXT CONSISTENCY ANALYSIS", ["Metric", "Value", "Interpretation"], max_rows=10)
    add_figure(doc, ART / "clip_similarity_by_class.png", "Fig. 6. CLIP similarity by FakeNewsNet label.")
    add_figure(doc, ART / "clip_inconsistency_roc_curve.png", "Fig. 7. CLIP inconsistency ROC curve.")
    add_para(
        doc,
        "The key interpretation is that visual consistency and factual truth are different. A fake article can use a semantically matching celebrity image, producing high CLIP similarity. A real article can use a generic or cropped image, producing lower similarity. Therefore, CLIP is useful for qualitative alignment analysis and failure-case grouping, but the project still requires a text fake-news branch."
    )
    add_figure(doc, ART / "fake_high_clip_similarity_grid.png", "Fig. 8. Fake articles with high CLIP similarity demonstrate why CLIP alone is insufficient.")
    add_figure(doc, ART / "fake_low_clip_similarity_grid.png", "Fig. 9. Fake articles with low CLIP similarity are plausible image-text inconsistency cases.")

    add_heading(doc, "VII. Discussion and Error Analysis", 1)
    add_para(
        doc,
        "The main strength of the final system is the experimental protocol: the same clean split is used across models, valid-image-only comparisons are separated from text-only comparisons, source/domain fields are not used as predictors, and leakage checks are performed before final claims. The main weakness is that FakeNewsNet labels factuality rather than direct image-text inconsistency. As a result, even rich visual-language representations do not necessarily improve fake/real classification."
    )
    if not failure_summary.empty:
        add_df_table(doc, failure_summary, "TABLE V. FAILURE-CASE CATEGORY SUMMARY", ["failure_category", "count"])
    add_para(
        doc,
        "Per-source evaluation shows severe imbalance: the final test split contains 509 GossipCop rows and only 9 PolitiFact rows. PolitiFact metrics are therefore diagnostic rather than reliable. The project reports this limitation rather than overgeneralizing from a small source subset."
    )
    add_figure(doc, "plot_17_domain_wise_macro_f1.png", "Fig. 10. Per-source/domain macro-F1 diagnostic plot.")

    add_heading(doc, "VIII. Engineering Challenges Solved", 1)
    add_bullets(
        doc,
        [
            "Image-content leakage: SHA-256 hashing revealed repeated image content across splits. Validation/test rows with prior-split image hashes were removed to produce image-deleaked split files.",
            "Fast overfitting: early deep models overfit within a few epochs. The final workflow added early stopping, dropout, weight decay, label smoothing, lower learning rates, and validation macro-F1 model selection.",
            "Split mismatch: several earlier artifacts used a 540-row test split, while the de-leaked final test split has 518 rows. Final text baselines were recomputed and report tables were rewired to use the aligned 518-row split.",
            "Pretrained model availability: CLIP, BERT, BLIP, ResNet50, and EasyOCR depend on local model caches and package availability. Optional cells now skip cleanly or cache outputs, and generated features are stored for reproducibility.",
            "Article-body extraction: historical URLs often fail or return short/blocked pages. The final content pipeline preserves all rows and uses `text_for_model` to fall back to title when full content is unavailable.",
            "Conceptual label mismatch: the dataset labels fake/real articles, not consistency/inconsistency. The final report separates CLIP proxy consistency analysis from supervised fake-news classification.",
        ],
    )

    add_heading(doc, "IX. Reproducibility and Repository Organization", 1)
    add_para(
        doc,
        "The repository is organized into notebooks and generated artifacts. Notebook 01 performs EDA, Notebook 02 prepares images, Notebook 03 trains and evaluates the final models, Notebook 04 audits dataset credibility, Notebook 05 runs the rich multimodal branch, Notebook 06 extracts article content, and Notebook 07 builds content/title modeling folders. Report-ready CSV files and plots are saved in `final_artifacts/`."
    )
    add_para(doc, f"Active GitHub repository: {GITHUB_URL}")
    add_para(
        doc,
        "To reproduce the headline numbers, run the split-preparation notebooks, then Notebook 03 final sections. Rich OCR/BLIP/BERT/ResNet50 results can be reproduced from Notebook 05; when cached features exist, reruns are fast. The content-enriched split folders can be rebuilt by running Notebook 06 followed by Notebook 07."
    )

    add_heading(doc, "X. Timeline and Task Assignment", 1)
    timeline = pd.DataFrame(
        [
            ["June 1-3, 2026", "Dataset EDA, title cleaning, source/domain audit", "Mostafa + Mohamed Hasan"],
            ["June 4-5, 2026", "Image download, image validation, image manifests", "Mohamed Ghanem"],
            ["June 6-7, 2026", "Text-only TF-IDF and transformer baselines", "Mostafa"],
            ["June 8, 2026", "CNN, concat fusion, consistency fusion, ResNet50 fusion", "Mostafa + Mohamed Ghanem"],
            ["June 9, 2026", "Leakage checks, de-leaked splits, valid-image-only evaluation", "Mohamed Hasan + Mostafa"],
            ["June 10, 2026", "CLIP consistency analysis, late fusion, failure-case tables", "All members"],
            ["June 11, 2026", "OCR/BLIP/BERT/ResNet50 trial, content extraction folders, final report", "All members"],
        ],
        columns=["Date", "Task", "Responsible"],
    )
    add_df_table(doc, timeline, "TABLE VI. FINAL PROJECT TIMELINE WITH DATES AND ASSIGNMENTS")

    add_heading(doc, "XI. Team Contributions", 1)
    add_para(
        doc,
        "Mohamed Hasan: coordinated final project scope, interpreted rubric feedback, contributed to the distinction between fake/real factuality and image-text consistency, reviewed leakage and source-bias risks, helped organize the report and presentation narrative, and supported final analysis writing."
    )
    add_para(
        doc,
        "Mostafa Elofy: implemented the main modeling workflow, including TF-IDF Logistic Regression, transformer text baseline, CLIP evaluation, late fusion, report-ready tables, notebook organization, artifact generation, and final report automation. He also maintained the repository structure and reproducibility scripts."
    )
    add_para(
        doc,
        "Mohamed Ghanem: focused on image-side work, including image validation, CNN image branch, frozen ResNet50 fusion, image-only ablation, rich OCR/BLIP/BERT/ResNet50 trial support, and visual failure-case inspection for the presentation."
    )

    add_heading(doc, "XII. GenAI Disclosure", 1)
    add_para(
        doc,
        "Generative AI assistance was used as a coding and writing support tool for notebook organization, debugging, report drafting, and wording refinement. All experiments, model outputs, metrics, tables, and figures reported in this document are based on project artifacts generated in the local repository. GenAI was not used to fabricate results, alter labels, or claim unavailable experiments. The team reviewed and selected the final interpretations and remains responsible for the submitted content."
    )

    add_heading(doc, "XIII. Conclusion", 1)
    add_para(
        doc,
        "The final project demonstrates that a rigorous multimodal fake-news pipeline requires more than adding images to text. On the final de-leaked valid-image split, text remains the strongest signal, and the best multimodal improvements are small. CLIP inconsistency analysis reveals that image-title alignment is not equivalent to factual truth, while image-only and rich visual-language trials show that heavier features can add noise. The strongest contribution is therefore a transparent evaluation framework: multiple baselines, leakage checks, valid-image-only evaluation, source analysis, and failure-case interpretation. These results are scientifically useful because they prevent overclaiming and clarify what would be required for a publication-level dataset: direct image-text consistency labels, stronger article-content extraction, and more balanced source coverage."
    )

    add_heading(doc, "References", 1)
    refs = [
        'H. Rashkin, E. Choi, J. Y. Jang, S. Volkova, and Y. Choi, "Truth of Varying Shades: Analyzing Language in Fake News and Political Fact-Checking," in Proc. EMNLP, 2017.',
        'Z. Jin, J. Cao, H. Guo, Y. Zhang, and J. Luo, "Multimodal Fusion with Co-Attention Networks for Fake News Detection," in Proc. ACL, 2017.',
        'S. Khattar, J. S. Goud, M. Gupta, and V. Varma, "MVAE: Multimodal Variational Autoencoder for Fake News Detection," in Proc. WWW Companion, 2019.',
        'K. Shu, D. Mahudeswaran, S. Wang, D. Lee, and H. Liu, "FakeNewsNet: A Data Repository with News Content, Social Context, and Spatiotemporal Information for Studying Fake News on Social Media," Big Data, 2020.',
        'J. Devlin, M.-W. Chang, K. Lee, and K. Toutanova, "BERT: Pre-training of Deep Bidirectional Transformers for Language Understanding," in Proc. NAACL-HLT, 2019.',
        'A. Radford et al., "Learning Transferable Visual Models From Natural Language Supervision," in Proc. ICML, 2021.',
        'J. Li et al., "BLIP: Bootstrapping Language-Image Pre-training for Unified Vision-Language Understanding and Generation," in Proc. ICML, 2022.',
        'K. He, X. Zhang, S. Ren, and J. Sun, "Deep Residual Learning for Image Recognition," in Proc. CVPR, 2016.',
        'K. Shu, S. Wang, and H. Liu, "Beyond News Contents: The Role of Social Context for Fake News Detection," in Proc. WSDM, 2019.',
        'Y. Zhou and R. Zafarani, "A Survey of Fake News: Fundamental Theories, Detection Methods, and Opportunities," ACM Computing Surveys, 2020.',
    ]
    for i, ref in enumerate(refs, start=1):
        add_para(doc, f"[{i}] {ref}")

    add_heading(doc, "Appendix: Report Artifacts", 1)
    add_bullets(
        doc,
        [
            "Core final notebook: 03_Multimodal_Model_Training_FINAL_Enhanced.ipynb.",
            "Rich branch runner: 05_Run_Rich_Multimodal_Branch.ipynb.",
            "Article content extraction: 06_Article_Content_Extraction_and_Content_Splits.ipynb.",
            "Content/title folder builder: 07_Build_Content_Title_Modeling_Splits.ipynb.",
            "Main result table: final_artifacts/report_table_classification_results.csv.",
            "CLIP consistency table: final_artifacts/report_table_clip_consistency.csv.",
            "Failure cases and diagnostic groups: final_artifacts/group_A-D_*.csv and figure grids.",
        ],
    )

    doc.save(OUT)
    print(f"Saved {OUT.resolve()}")


if __name__ == "__main__":
    main()
