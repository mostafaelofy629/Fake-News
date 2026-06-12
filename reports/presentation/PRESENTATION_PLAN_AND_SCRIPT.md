# Project 11 Presentation Plan: Multimodal Fake News Detection

Target length: 18-20 minutes  
Deck style: keep the yellow/black visual style from `Yellow and Black Modern True Crime Presentation.pdf`, but shift the mood from "true crime" to "digital evidence investigation".

## Core Message

Project 11 is **Detecting Inconsistency Between Image and Text in News Articles**. The implementation uses FakeNewsNet fake/real classification as the supervised proxy while the modeling motivation is image-text consistency.

This project investigated whether combining article titles and images improves fake-news classification. The strongest lesson is not simply "multimodal is better"; it is more careful:

> After leakage checks and image de-duplication, text-only and multimodal fusion models are very close. Images alone are weak, but multimodal fusion is competitive and diagnostically useful. The biggest contribution is building a credible multimodal evaluation pipeline rather than trusting a noisy dataset blindly.

Do not overclaim that the multimodal model clearly beats text-only. The best multimodal consistency fusion is close to the best text baselines, but statistical tests do not show a decisive win.

## Instruction And Report Alignment

The presentation should explicitly show that Project 11 requirements were covered:

| Required component | Evidence to show |
|---|---|
| Image + text data pipeline | Notebook 01 EDA, Notebook 02 image preparation, valid image manifests |
| Text-only baseline | TF-IDF + Logistic Regression, optional DistilBERT |
| Multimodal model | CNN/ResNet image branch + text branch + fusion classifier |
| Evaluation | Accuracy, macro-F1, confusion matrices, bootstrap CI, McNemar tests |
| Qualitative analysis | Failure cases, image/text inspection, source-bias categories |
| Reproducibility | Saved splits, checkpoints, CSV predictions, artifacts, README |

The report also emphasizes these risks: stale image URLs, class imbalance, source/domain shortcut learning, limited compute, and overfitting. Keep these as a named section in the presentation.

## What To Change In The Existing PDF

1. Keep the black/yellow theme.
2. Replace any generic "true crime" language with "fake news investigation", "evidence", "leakage", "bias", "model audit", and "cross-modal clues".
3. Add more project-specific plots and tables:
   - `plot_01_label_distribution.png`
   - `plot_08_image_coverage_source_label.png`
   - `plot_11_model_architecture.png`
   - `plot_13_training_curves.png`
   - `plot_16_final_model_comparison.png`
   - `final_artifacts/final_image_content_leakage_pair_summary.csv`
   - `final_artifacts/final_deep_ablation_results_mean_std.csv`
   - `final_artifacts/final_mcnemar_tests.csv`
   - `final_artifacts/final_failure_category_summary.csv`
4. Remove slides that only describe fake news generally if they do not connect to the experiment.
5. Add one "hard truth" slide: multimodal did not strongly beat text-only after cleaning.
6. Add one "hurdles and fixes" slide. This will make the project look mature and honest.

## Recommended Slide Order

### Slide 1: 10-Second Opening Video

Visual:
- A fast sequence: social media feed -> suspicious headline -> image mismatch -> model pipeline -> final verdict.
- Use yellow highlight boxes over black background.
- End frame text: "Can a model detect fake news from both words and images?"

On-slide text:
- No more than one sentence:
  "Fake news is not only written. It is staged, packaged, and amplified."

Speaker script:
> We start with a simple idea: misinformation is rarely just text. It often arrives with an image, a headline, and a source. Our project asks whether a model can learn from both the words and the visual evidence.

Time: 0:10

### Slide 2: Grabbing Intro

Title:
The Case: When Text And Images Become Evidence

Content:
- Fake-news posts often combine title, image, and source context.
- A text-only model may miss visual inconsistency.
- A multimodal model may detect patterns across both.

Speaker script:
> Our project treats each news item like a small evidence packet: a title, an image, a source, and a fake-or-real label. The question is not only whether we can classify fake news, but whether the image adds independent value beyond the title.

Time: 1:00

### Slide 3: Research Question

Title:
Project 11 Research Question

Content:
- Can image + text improve fake/real news classification?
- Which modality carries more useful signal?
- Can we trust the evaluation split?
- How well does this proxy task support the original goal: detecting image-text inconsistency?

Speaker script:
> Project 11 asks us to detect inconsistency between image and text in news articles. FakeNewsNet gives us fake and real labels, so we use fake/real classification as the measurable proxy. We framed the work around four questions: does multimodal fusion help, which modality carries the signal, can we trust the split, and how much does this proxy actually reflect image-text inconsistency?

Time: 1:10

### Slide 4: Requirement Coverage

Title:
How We Covered Project 11

Content:
- Image + text pipeline
- Text-only baseline
- Multimodal fusion model
- Failure-case analysis
- Reproducible artifacts

Speaker script:
> The official project requirements map directly to our notebooks. Notebook 01 handles EDA and split checks. Notebook 02 prepares image files and multimodal CSVs. Notebook 03 trains text-only and multimodal models. Notebook 04 contains the deeper dataset credibility audit. The outputs are saved as plots, CSVs, predictions, checkpoints, and report-ready tables.

Time: 1:00

### Slide 5: Dataset

Title:
Dataset: FakeNewsNet

Content:
- Sources: GossipCop and PolitiFact
- Labels: fake / real
- Modalities: article title + downloaded image
- Original data had strong class and source imbalance.
- Important limitation: fake/real is used as the supervised label, not direct image-text inconsistency.

Speaker script:
> We used FakeNewsNet, mainly GossipCop and PolitiFact records. The original data is imbalanced: the report found many more real than fake articles, and GossipCop dominates the sample count. The labels are fake and real, so our model is not directly trained on "image-text mismatch"; instead, it learns fake-news classification using both title and image features.

Time: 1:10

### Slide 6: Data Pipeline

Title:
From Raw Records To Model-Ready Splits

Content:
- Clean metadata and labels
- Download and validate images
- Build train / validation / test splits
- Create valid-image-only subset
- Save audit artifacts

Speaker script:
> Before modeling, we built the pipeline: clean the CSV metadata, download images, verify image paths, create train-validation-test splits, and generate audit files. This was important because missing images and duplicated content can make multimodal results misleading.

Time: 1:15

### Slide 7: First Hurdle: Image Availability Bias

Title:
Problem 1: Not Every Article Has A Valid Image

Content:
- Image availability differed by source and label.
- Placeholder images created a risk of shortcut learning.
- Fix: use valid-image-only results for headline multimodal claims.

Suggested visual:
- `plot_08_image_coverage_source_label.png`

Speaker script:
> Our first major problem was image coverage. Not every article had a valid downloaded image, and coverage was not equal across source and label. If we trained with placeholders, the model could learn missingness instead of news evidence. So we kept placeholder runs as diagnostic only and used valid-image-only splits for the main multimodal claims.

Time: 1:20

### Slide 8: Second Hurdle: Leakage

Title:
Problem 2: Clean IDs Were Not Enough

Content:
- ID overlap: clean
- Normalized title overlap: clean
- Image-content overlap: not clean
- Exact image duplicates appeared across splits.

Key numbers:
- train-val overlapping image hashes: 22
- train-test overlapping image hashes: 13
- val-test overlapping image hashes: 6

Speaker script:
> We initially checked ID and title leakage, and those looked clean. But then we checked image content itself. That revealed duplicate images across splits. This matters because a model can recognize an image it has already seen, even if the article ID is different.

Time: 1:20

### Slide 9: How We Fixed Leakage

Title:
De-Leaking The Dataset

Content:
- Kept train unchanged
- Removed validation images already seen in train
- Removed test images already seen in train or cleaned validation

Key numbers:
- validation: 566 -> 541
- test: 540 -> 518

Speaker script:
> Our fix was conservative. Training stayed unchanged. Validation rows were removed if their image content appeared in training. Test rows were removed if their image appeared in training or cleaned validation. This gave us a cleaner, more honest evaluation.

Time: 1:15

### Slide 10: Modeling Strategy

Title:
Models We Compared

Content:
- Majority baseline
- TF-IDF + Logistic Regression
- DistilBERT text-only
- Image-only CNN
- CNN + text concat fusion
- CNN + text consistency fusion
- Frozen ResNet50 + text fusion

Speaker script:
> We did not only train one model. We built a ladder of baselines and ablations: majority baseline, classical text, pretrained text, image-only, custom CNN fusion, consistency fusion, and frozen ResNet50 fusion. This lets us ask where the performance is really coming from.

Time: 1:20

### Slide 11: Architecture

Title:
Multimodal Fusion Architecture

Content:
- Text branch: TF-IDF dense features or pretrained text baseline
- Image branch: CNN or frozen ResNet50
- Fusion: concatenate text, image, absolute difference, and element-wise product
- Classifier: fake / real

Suggested visual:
- `plot_11_model_architecture.png`

Speaker script:
> The main fusion model projects text and image features into the same latent space. Then it uses not only concatenation, but also absolute difference and element-wise product. Those extra terms are a simple way to represent agreement and disagreement between modalities.

Time: 1:20

### Slide 12: Training Improvements

Title:
Training Against Overfitting

Content:
- Weighted cross-entropy
- AdamW optimizer
- Lower learning rate
- Stronger dropout
- Label smoothing
- Early stopping
- Multiple seeds

Speaker script:
> Early training curves showed fast overfitting. Validation performance often peaked early while training performance continued to rise. We responded with AdamW, lower learning rates, stronger dropout, label smoothing, and early stopping based on validation macro-F1. We also repeated final deep models across three seeds.

Time: 1:10

### Slide 13: Results

Title:
Main Results After Cleaning

Content:
Use a compact table:

| Model | Mean Macro-F1 |
|---|---:|
| Consistency fusion CNN+Text | 0.7608 |
| TF-IDF Logistic Regression | 0.7576 |
| Concat fusion CNN+Text | 0.7575 |
| DistilBERT text-only | 0.7511 |
| Frozen ResNet50 fusion | 0.7421 |
| Image-only CNN | 0.5437 |

Speaker script:
> After cleaning, the strongest models are very close. Consistency fusion is the highest mean macro-F1 at about 0.761, but TF-IDF logistic regression and concat fusion are only slightly lower. DistilBERT is useful as a pretrained text baseline, but in the latest saved run it is not the winner. Image-only performance is much weaker, which tells us that images alone do not carry enough reliable signal in this dataset.

Time: 1:40

### Slide 14: Statistical Interpretation

Title:
Did Multimodal Clearly Beat Text?

Content:
- Bootstrap confidence intervals overlap.
- McNemar tests did not show a decisive improvement over text-only.
- Image-only is significantly worse.

Speaker script:
> The honest answer is: not clearly. The multimodal models are competitive with text-only models, but the confidence intervals overlap and McNemar tests do not show a decisive win against text. The only clear result is that image-only is much weaker.

Time: 1:20

### Slide 15: Error Analysis

Title:
Where The Model Still Fails

Content:
- Most failures require source/domain bias checks.
- Some titles are short or ambiguous.
- Fake-news labels may not correspond to visible image inconsistency.

Key artifact:
- `final_artifacts/final_failure_category_summary.csv`

Speaker script:
> The failure analysis showed that many errors are not simple visual mistakes. A large group needs source and domain bias checking, and some titles are too short or ambiguous. Also, a fake article can use a real-looking image, so the image may not contradict the text directly.

Time: 1:20

### Slide 16: What We Learned

Title:
What The Evidence Says

Content:
- Text is very strong in this dataset.
- Images are noisy and often duplicated.
- Multimodal fusion is competitive, not clearly superior.
- Data credibility checks changed the story.

Speaker script:
> The biggest lesson is that a better architecture is not enough. Without leakage checks, multimodal results can look stronger than they really are. After cleaning, text and multimodal fusion are very close, while image-only remains weak. So the project becomes a careful, credible evaluation rather than a simple claim that more modalities automatically win.

Time: 1:20

### Slide 17: Team Contributions And Reproducibility

Title:
How The Work Was Divided

Content:
- Mohamed Ghanem: image preparation, validation, CNN/image branch, visual inspection
- Mostafa Mohie: text cleaning, TF-IDF/DistilBERT baselines, notebook utilities
- Mohamed Kamal Hasan: splitting, fusion model, evaluation, final integration
- Shared: debugging, documentation, presentation

Speaker script:
> The work was divided by modality and evaluation. One part focused on image preparation and visual features, one part on text processing and baselines, and one part on split construction, fusion, and evaluation. Everyone contributed to debugging, documentation, and final presentation preparation.

Time: 0:50

### Slide 18: Limitations And Future Work

Title:
Limitations And Next Steps

Content:
- Use CLIP for image-text alignment
- Add OCR from images
- Include article body, not only title
- Evaluate on a dataset labeled for image-text inconsistency
- Expand beyond GossipCop-heavy distribution

Speaker script:
> For future work, CLIP is the most natural next step because it is trained for image-text alignment. OCR could capture text inside images. Article body text would also provide richer context. Finally, a dataset labeled directly for image-text inconsistency would better match the multimodal objective.

Time: 1:20

### Slide 19: Closing

Title:
Final Takeaway

Content:
> The project did not just train a multimodal model. It audited whether the multimodal evidence was trustworthy.

Speaker script:
> Our final takeaway is that fake-news detection is not only a modeling problem. It is also a data credibility problem. We built the models, but more importantly, we tested whether the evidence was clean enough to trust.

Time: 0:40

## Timing Summary

| Section | Slides | Time |
|---|---:|---:|
| Hook and question | 1-3 | 2:20 |
| Requirement and dataset credibility | 4-9 | 7:20 |
| Modeling | 10-12 | 3:50 |
| Results and interpretation | 13-15 | 4:20 |
| Team, future work, closing | 16-19 | 3:50 |
| Total | 19 slides | about 19:40 |

## Hurdles And Solutions To Mention

### Hurdle 1: Missing or uneven image coverage

Problem:
- Not all articles had valid downloaded images.
- Coverage differed by source and label.

Solution:
- Treat placeholder-image runs as diagnostic only.
- Use valid-image-only splits for headline multimodal claims.

### Hurdle 2: Data leakage was hidden in images

Problem:
- ID and title overlap checks were clean.
- But identical image content appeared across train, validation, and test.

Solution:
- Add image hash audit.
- Remove validation/test rows whose image appeared in earlier splits.
- Create de-leaked split files.

### Hurdle 3: Fast overfitting

Problem:
- Training macro-F1 rose while validation macro-F1 peaked early.

Solution:
- Early stopping.
- AdamW and weight decay.
- Lower learning rates.
- Stronger dropout.
- Label smoothing.

### Hurdle 4: Multimodal did not clearly outperform text

Problem:
- The image signal was weaker than expected.

Solution:
- Report the result honestly.
- Compare against strong text baselines.
- Use ablations to show where signal comes from.

### Hurdle 5: Dataset label is not true multimodal inconsistency

Problem:
- Fake/real labels do not always mean image-text mismatch.

Solution:
- Frame the task as fake/real classification using multimodal evidence.
- Recommend CLIP and inconsistency-labeled datasets for future work.

## Design Notes For The Yellow/Black Deck

Use these section labels:

- "The Case"
- "The Evidence"
- "The Suspects: Leakage and Bias"
- "The Models"
- "The Verdict"
- "What We Learned"

Visual style:
- Black background.
- Yellow numbers for key metrics.
- White body text.
- Use red sparingly only for warnings: leakage, bias, overfitting.
- Use screenshots/plots as evidence cards.

Avoid:
- Too much code.
- Giant tables.
- Claiming multimodal is clearly better.
- Spending more than 2 minutes on architecture details.

## Short Opening Video Storyboard

Length: 10 seconds

0-2 seconds:
- Rapid feed of headlines/images.
- Text overlay: "Every post tells a story..."

2-4 seconds:
- Highlight title and image separately.
- Overlay: "But do they agree?"

4-6 seconds:
- Split screen: text branch and image branch.
- Overlay: "Text + Image"

6-8 seconds:
- Warning stamps: "Missing images", "Duplicates", "Leakage"
- Overlay: "First, audit the evidence."

8-10 seconds:
- Final model verdict screen.
- Overlay: "Multimodal Fake News Detection"

## One-Sentence Answers For Q&A

Q: Why did image-only perform poorly?  
A: The images are noisy, duplicated, and not always semantically tied to fake/real labels.

Q: Why use valid-image-only splits?  
A: Because placeholder images can teach the model missingness patterns instead of visual evidence.

Q: Did multimodal beat text-only?  
A: Consistency fusion had the best mean macro-F1, but it was not decisively better than text-only after leakage cleaning.

Q: Why is this called image-text inconsistency if the labels are fake/real?  
A: FakeNewsNet provides fake/real labels, so we use them as a supervised proxy while the architecture and failure analysis focus on cross-modal evidence.

Q: What was the most important technical fix?  
A: Image-content leakage detection and de-leaked evaluation splits.

Q: What would improve the project next?  
A: CLIP embeddings, OCR, article-body text, and a dataset labeled for image-text inconsistency.

## Final Recommendation

Use the project as a story of scientific honesty:

1. We expected multimodal learning to help.
2. The raw data had hidden risks.
3. We audited and cleaned those risks.
4. After cleaning, text remained very strong and multimodal fusion was only slightly ahead.
5. Multimodal fusion was competitive, but the real contribution was credible evaluation.
