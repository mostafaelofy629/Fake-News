import nbformat
from nbformat.v4 import new_markdown_cell, new_code_cell
from pathlib import Path

nb_path = Path("03_Multimodal_Model_Training_FINAL_Enhanced.ipynb")
nb = nbformat.read(nb_path, as_version=4)

marker = "FINAL DIAGNOSTIC: Training-Set Errors Across Saved Models"
if any(marker in cell.get("source", "") for cell in nb.cells):
    print("Training-set diagnostic already exists; no append needed.")
else:
    nb.cells.append(new_markdown_cell(f"""## {marker}

This diagnostic evaluates the saved trained models on the **training split** instead of the test split. It answers:

> How many training samples are still misclassified by more than 6 model runs?

This is not a generalization metric. It is a memorization/fit diagnostic: if many training samples are still repeatedly wrong, those records may be noisy, mislabeled, visually ambiguous, or outside the learned pattern."""))

    nb.cells.append(new_code_cell(r'''# FINAL DIAGNOSTIC: evaluate saved models on the training split and count samples wrong in >6 model runs
from pathlib import Path
import re
import numpy as np
import pandas as pd

TRAIN_WRONG_THRESHOLD = 6
FINAL_ARTIFACT_DIR = Path('final_artifacts')
DATA_SPLIT_DIR = Path('data_splits')
LABEL_TO_ID = {'real': 0, 'fake': 1}
ID_TO_LABEL = {0: 'real', 1: 'fake'}

def _diag_load_train_split():
    candidates = [
        DATA_SPLIT_DIR / 'train_valid_images_only_image_deleaked_with_content.csv',
        DATA_SPLIT_DIR / 'train_valid_images_only_image_deleaked.csv',
        DATA_SPLIT_DIR / 'train_valid_images_only.csv',
        DATA_SPLIT_DIR / 'train_multimodal.csv',
        DATA_SPLIT_DIR / 'train.csv',
    ]
    for path in candidates:
        if path.exists():
            df = pd.read_csv(path)
            df['split'] = 'train'
            if 'label_id' not in df.columns:
                df['label_id'] = df['label'].map(LABEL_TO_ID).astype(int)
            df['label_id'] = df['label_id'].astype(int)
            if 'valid_image_path' not in df.columns:
                img_col = 'image_path_primary' if 'image_path_primary' in df.columns else 'image_path'
                df['valid_image_path'] = df[img_col].fillna('').astype(str).map(lambda p: bool(p) and Path(p).exists()) if img_col in df.columns else False
            print('Training diagnostic loaded:', path, 'rows=', len(df))
            return df
    raise FileNotFoundError('No train split found.')

train_diag = _diag_load_train_split()

def _diag_metric_row(y_true, y_pred, prob_fake=None, model='model'):
    from sklearn.metrics import accuracy_score, f1_score, roc_auc_score, average_precision_score
    row = {
        'model': model,
        'n_samples': int(len(y_true)),
        'accuracy': accuracy_score(y_true, y_pred),
        'macro_f1': f1_score(y_true, y_pred, average='macro', zero_division=0),
    }
    if prob_fake is not None and len(np.unique(y_true)) == 2:
        row['roc_auc'] = roc_auc_score(y_true, prob_fake)
        row['pr_auc'] = average_precision_score(y_true, prob_fake)
    else:
        row['roc_auc'] = np.nan
        row['pr_auc'] = np.nan
    return row

def _diag_make_pred_frame(base_df, model_name, pred_id, prob_fake):
    out = base_df[['id', 'source', 'title', 'label', 'label_id', 'valid_image_path']].copy()
    if 'image_path_primary' in base_df.columns:
        out['image_path_primary'] = base_df['image_path_primary']
    out['model_name'] = model_name
    out['model_family'] = re.sub(r'_seed\d+$', '', model_name)
    out['pred_id'] = np.asarray(pred_id, dtype=int)
    out['pred_label'] = [ID_TO_LABEL[int(i)] for i in out['pred_id']]
    out['prob_fake'] = prob_fake
    out['correct'] = out['pred_id'].eq(out['label_id'].astype(int))
    out['is_wrong'] = ~out['correct']
    return out

train_pred_frames = []
train_metric_rows = []

# 1) Recreate the final TF-IDF LR baseline and evaluate it on its own training data.
try:
    from sklearn.pipeline import Pipeline
    from sklearn.feature_extraction.text import TfidfVectorizer
    from sklearn.linear_model import LogisticRegression

    text_col = next((c for c in ['text_for_model', 'content', 'title'] if c in train_diag.columns), 'title')
    y_train_diag = train_diag['label_id'].astype(int).values
    text_lr_diag = Pipeline([
        ('tfidf', TfidfVectorizer(lowercase=True, stop_words='english', ngram_range=(1, 2), min_df=2, max_df=0.95, max_features=20000)),
        ('clf', LogisticRegression(max_iter=1000, class_weight='balanced', solver='liblinear', random_state=42)),
    ])
    text_lr_diag.fit(train_diag[text_col].fillna('').astype(str), y_train_diag)
    pred = text_lr_diag.predict(train_diag[text_col].fillna('').astype(str))
    prob = text_lr_diag.predict_proba(train_diag[text_col].fillna('').astype(str))[:, 1]
    name = 'train_eval_text_tfidf_lr'
    train_metric_rows.append(_diag_metric_row(y_train_diag, pred, prob, name))
    train_pred_frames.append(_diag_make_pred_frame(train_diag, name, pred, prob))
    print('Evaluated:', name)
except Exception as exc:
    print('Skipped TF-IDF train diagnostic:', repr(exc))

# 2) Load saved PyTorch deep models and evaluate on training data.
try:
    import torch
    import torch.nn as nn
    from torch.utils.data import Dataset, DataLoader
    from torchvision import transforms, models
    from PIL import Image

    DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'
    IMAGE_SIZE = 224
    BATCH_SIZE = 32
    TEXT_DROPOUT = 0.40
    FUSION_DROPOUT = 0.50

    deep_vectorizer = text_lr_diag.named_steps['tfidf']
    # Match the original deep cell: saved fusion models were trained from title TF-IDF vectors.
    X_train_deep = deep_vectorizer.transform(train_diag['title'].fillna('').astype(str)).astype(np.float32).toarray()

    eval_transform = transforms.Compose([
        transforms.Resize((IMAGE_SIZE, IMAGE_SIZE)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])

    class TrainDiagDataset(Dataset):
        def __init__(self, df, text_array, transform=None):
            self.df = df.reset_index(drop=True)
            self.text_array = text_array
            self.transform = transform
        def __len__(self):
            return len(self.df)
        def __getitem__(self, idx):
            row = self.df.iloc[idx]
            img_path = row.get('image_path_primary', '') if 'image_path_primary' in row.index else row.get('image_path', '')
            valid = isinstance(img_path, str) and Path(img_path).exists()
            try:
                img = Image.open(img_path).convert('RGB') if valid else Image.new('RGB', (IMAGE_SIZE, IMAGE_SIZE), (128, 128, 128))
            except Exception:
                img = Image.new('RGB', (IMAGE_SIZE, IMAGE_SIZE), (128, 128, 128))
            if self.transform:
                img = self.transform(img)
            return (
                torch.tensor(self.text_array[idx], dtype=torch.float32),
                img,
                torch.tensor(int(row['label_id']), dtype=torch.long),
            )

    class SmallCNNEncoderFinal(nn.Module):
        def __init__(self, out_dim=128):
            super().__init__()
            self.features = nn.Sequential(
                nn.Conv2d(3, 32, 3, padding=1), nn.BatchNorm2d(32), nn.ReLU(), nn.MaxPool2d(2),
                nn.Conv2d(32, 64, 3, padding=1), nn.BatchNorm2d(64), nn.ReLU(), nn.MaxPool2d(2),
                nn.Conv2d(64, 128, 3, padding=1), nn.BatchNorm2d(128), nn.ReLU(), nn.AdaptiveAvgPool2d((1, 1)),
            )
            self.proj = nn.Linear(128, out_dim)
        def forward(self, x):
            return self.proj(self.features(x).flatten(1))

    class FrozenResNet50Encoder(nn.Module):
        def __init__(self):
            super().__init__()
            self.backbone = models.resnet50(weights=None)
            self.out_dim = self.backbone.fc.in_features
            self.backbone.fc = nn.Identity()
            for p in self.backbone.parameters():
                p.requires_grad = False
        def forward(self, image):
            self.backbone.eval()
            with torch.no_grad():
                return self.backbone(image)

    class ImageOnlyNet(nn.Module):
        def __init__(self, out_dim=128):
            super().__init__()
            self.image_encoder = SmallCNNEncoderFinal(out_dim)
            self.classifier = nn.Sequential(nn.ReLU(), nn.Dropout(0.35), nn.Linear(out_dim, 2))
        def forward(self, text, image):
            return self.classifier(self.image_encoder(image))

    class ConcatFusionNet(nn.Module):
        def __init__(self, text_in_dim, latent_dim=128):
            super().__init__()
            self.text_proj = nn.Sequential(nn.Linear(text_in_dim, latent_dim), nn.ReLU(), nn.Dropout(TEXT_DROPOUT))
            self.image_encoder = SmallCNNEncoderFinal(latent_dim)
            self.classifier = nn.Sequential(nn.Linear(latent_dim * 2, latent_dim), nn.ReLU(), nn.Dropout(FUSION_DROPOUT), nn.Linear(latent_dim, 2))
        def forward(self, text, image):
            t = self.text_proj(text)
            v = self.image_encoder(image)
            return self.classifier(torch.cat([t, v], dim=1))

    class ConsistencyFusionNetFinal(nn.Module):
        def __init__(self, text_in_dim, latent_dim=128):
            super().__init__()
            self.text_proj = nn.Sequential(nn.Linear(text_in_dim, latent_dim), nn.ReLU(), nn.Dropout(TEXT_DROPOUT))
            self.image_encoder = SmallCNNEncoderFinal(latent_dim)
            self.classifier = nn.Sequential(nn.Linear(latent_dim * 4, latent_dim), nn.ReLU(), nn.Dropout(FUSION_DROPOUT), nn.Linear(latent_dim, 2))
        def forward(self, text, image):
            t = self.text_proj(text)
            v = self.image_encoder(image)
            return self.classifier(torch.cat([t, v, torch.abs(t - v), t * v], dim=1))

    class FrozenResNet50FusionNet(nn.Module):
        def __init__(self, text_in_dim, latent_dim=128):
            super().__init__()
            self.image_encoder = FrozenResNet50Encoder()
            self.available = True
            self.text_proj = nn.Sequential(nn.Linear(text_in_dim, latent_dim), nn.ReLU(), nn.Dropout(TEXT_DROPOUT))
            self.image_proj = nn.Sequential(nn.Linear(self.image_encoder.out_dim, latent_dim), nn.ReLU(), nn.Dropout(0.35))
            self.classifier = nn.Sequential(nn.Linear(latent_dim * 4, latent_dim), nn.ReLU(), nn.Dropout(FUSION_DROPOUT), nn.Linear(latent_dim, 2))
        def forward(self, text, image):
            t = self.text_proj(text)
            v = self.image_proj(self.image_encoder(image))
            return self.classifier(torch.cat([t, v, torch.abs(t - v), t * v], dim=1))

    def _deep_ctor_for_path(path, text_dim):
        name = path.name.lower()
        if name.startswith('image-only'):
            return ImageOnlyNet()
        if name.startswith('concat_fusion'):
            return ConcatFusionNet(text_dim)
        if name.startswith('consistency_fusion'):
            return ConsistencyFusionNetFinal(text_dim)
        if name.startswith('frozen_resnet50'):
            return FrozenResNet50FusionNet(text_dim)
        return None

    def _eval_torch_model(model, loader):
        model.eval()
        ys, preds, probs = [], [], []
        with torch.no_grad():
            for text, image, y in loader:
                text, image = text.to(DEVICE), image.to(DEVICE)
                logits = model(text, image)
                prob = torch.softmax(logits, dim=1)[:, 1].detach().cpu().numpy()
                pred = logits.argmax(dim=1).detach().cpu().numpy()
                ys.extend(y.numpy().tolist())
                preds.extend(pred.tolist())
                probs.extend(prob.tolist())
        return np.asarray(ys), np.asarray(preds), np.asarray(probs)

    train_loader_diag = DataLoader(TrainDiagDataset(train_diag, X_train_deep, eval_transform), batch_size=BATCH_SIZE, shuffle=False, num_workers=0)
    deep_weight_files = sorted([
        *FINAL_ARTIFACT_DIR.glob('image-only_cnn_seed*.pt'),
        *FINAL_ARTIFACT_DIR.glob('concat_fusion_cnn+text_seed*.pt'),
        *FINAL_ARTIFACT_DIR.glob('consistency_fusion_cnn+text_seed*.pt'),
        *FINAL_ARTIFACT_DIR.glob('frozen_resnet50_fusion_cnn+text_seed*.pt'),
    ])
    for weight_path in deep_weight_files:
        model = _deep_ctor_for_path(weight_path, X_train_deep.shape[1])
        if model is None:
            continue
        ckpt = torch.load(weight_path, map_location=DEVICE, weights_only=False)
        state = ckpt.get('model_state_dict', ckpt) if isinstance(ckpt, dict) else ckpt
        model.load_state_dict(state, strict=True)
        model = model.to(DEVICE)
        y_true, pred, prob = _eval_torch_model(model, train_loader_diag)
        model_name = weight_path.stem
        train_metric_rows.append(_diag_metric_row(y_true, pred, prob, model_name))
        train_pred_frames.append(_diag_make_pred_frame(train_diag, model_name, pred, prob))
        print('Evaluated:', model_name)
except Exception as exc:
    print('Skipped saved deep-model train diagnostic:', repr(exc))

# 3) Optional rich cached-feature MLP on train, if state/features exist.
try:
    import torch
    import torch.nn as nn
    from torch.utils.data import DataLoader, TensorDataset
    from sklearn.preprocessing import StandardScaler

    rich_state_path = FINAL_ARTIFACT_DIR / 'rich_multimodal_mlp_best_state.pt'
    rich_feat_path = FINAL_ARTIFACT_DIR / 'rich_multimodal_train_features.npz'
    rich_meta_path = FINAL_ARTIFACT_DIR / 'rich_multimodal_train_ocr_blip_text.csv'
    if rich_state_path.exists() and rich_feat_path.exists() and rich_meta_path.exists():
        z = np.load(rich_feat_path)
        meta = pd.read_csv(rich_meta_path)
        X = z['features'].astype(np.float32)
        y = z['labels'].astype(int)
        scaler = StandardScaler()
        X = scaler.fit_transform(X).astype(np.float32)

        class RichMLPDiag(nn.Module):
            def __init__(self, dim):
                super().__init__()
                self.net = nn.Sequential(
                    nn.Linear(dim, 512), nn.BatchNorm1d(512), nn.ReLU(), nn.Dropout(0.45),
                    nn.Linear(512, 128), nn.BatchNorm1d(128), nn.ReLU(), nn.Dropout(0.45),
                    nn.Linear(128, 2),
                )
            def forward(self, x):
                return self.net(x)

        DEVICE_RICH = 'cuda' if torch.cuda.is_available() else 'cpu'
        rich_model = RichMLPDiag(X.shape[1]).to(DEVICE_RICH)
        state = torch.load(rich_state_path, map_location=DEVICE_RICH, weights_only=False)
        rich_model.load_state_dict(state, strict=True)
        rich_model.eval()
        loader = DataLoader(TensorDataset(torch.tensor(X), torch.tensor(y, dtype=torch.long)), batch_size=128, shuffle=False)
        ys, preds, probs = [], [], []
        with torch.no_grad():
            for xb, yb in loader:
                xb = xb.to(DEVICE_RICH)
                logits = rich_model(xb)
                probs.extend(torch.softmax(logits, dim=1)[:, 1].cpu().numpy().tolist())
                preds.extend(logits.argmax(dim=1).cpu().numpy().tolist())
                ys.extend(yb.numpy().tolist())
        model_name = 'rich_multimodal_train_cached_mlp'
        # meta is already in train split order for the cached rich features.
        meta_eval = meta.copy()
        if 'label_id' not in meta_eval.columns:
            meta_eval['label_id'] = meta_eval['label'].map(LABEL_TO_ID).astype(int)
        train_metric_rows.append(_diag_metric_row(np.asarray(ys), np.asarray(preds), np.asarray(probs), model_name))
        train_pred_frames.append(_diag_make_pred_frame(meta_eval, model_name, np.asarray(preds), np.asarray(probs)))
        print('Evaluated:', model_name)
    else:
        print('Rich train diagnostic skipped: cached rich train features/state are missing.')
except Exception as exc:
    print('Skipped rich train diagnostic:', repr(exc))

if not train_pred_frames:
    raise RuntimeError('No training-set predictions were produced.')

train_all_votes = pd.concat(train_pred_frames, ignore_index=True)
train_wrong_votes = train_all_votes[train_all_votes['is_wrong']].copy()

base_meta = (
    train_all_votes
    .sort_values(['id', 'title'], na_position='last')
    .groupby('id', as_index=False)
    .agg(
        source=('source', 'first'),
        title=('title', 'first'),
        label=('label', 'first'),
        total_train_model_runs=('model_name', 'nunique'),
        train_model_families_evaluated=('model_family', 'nunique'),
    )
)

wrong_summary = train_wrong_votes.groupby('id').agg(
    wrong_train_run_count=('model_name', 'nunique'),
    wrong_train_family_count=('model_family', 'nunique'),
    train_models_wrong=('model_name', lambda s: '; '.join(sorted(set(map(str, s))))),
    train_families_wrong=('model_family', lambda s: '; '.join(sorted(set(map(str, s))))),
).reset_index()

train_failure_summary = base_meta.merge(wrong_summary, on='id', how='left')
train_failure_summary['wrong_train_run_count'] = train_failure_summary['wrong_train_run_count'].fillna(0).astype(int)
train_failure_summary['wrong_train_family_count'] = train_failure_summary['wrong_train_family_count'].fillna(0).astype(int)
train_failure_summary['train_models_wrong'] = train_failure_summary['train_models_wrong'].fillna('')
train_failure_summary['train_families_wrong'] = train_failure_summary['train_families_wrong'].fillna('')
train_failure_summary['wrong_train_run_rate'] = train_failure_summary['wrong_train_run_count'] / train_failure_summary['total_train_model_runs'].replace(0, np.nan)
train_failure_summary['wrong_train_family_rate'] = train_failure_summary['wrong_train_family_count'] / train_failure_summary['train_model_families_evaluated'].replace(0, np.nan)
train_failure_summary = train_failure_summary.sort_values(
    ['wrong_train_run_count', 'wrong_train_family_count', 'wrong_train_run_rate'],
    ascending=[False, False, False],
).reset_index(drop=True)

train_metrics = pd.DataFrame(train_metric_rows).sort_values(['macro_f1', 'accuracy'], ascending=False).reset_index(drop=True)

n_wrong_gt6_runs = int((train_failure_summary['wrong_train_run_count'] > TRAIN_WRONG_THRESHOLD).sum())
n_wrong_ge6_runs = int((train_failure_summary['wrong_train_run_count'] >= TRAIN_WRONG_THRESHOLD).sum())
n_wrong_gt6_families = int((train_failure_summary['wrong_train_family_count'] > TRAIN_WRONG_THRESHOLD).sum())

train_all_votes.to_csv(FINAL_ARTIFACT_DIR / 'train_eval_all_model_votes_long.csv', index=False)
train_wrong_votes.to_csv(FINAL_ARTIFACT_DIR / 'train_eval_wrong_votes_long.csv', index=False)
train_failure_summary.to_csv(FINAL_ARTIFACT_DIR / 'train_eval_most_frequent_wrong_samples.csv', index=False)
train_metrics.to_csv(FINAL_ARTIFACT_DIR / 'train_eval_model_metrics.csv', index=False)

print(f'Training samples evaluated: {train_failure_summary["id"].nunique()}')
print(f'Training model runs evaluated: {train_all_votes["model_name"].nunique()}')
print(f'Training model families evaluated: {train_all_votes["model_family"].nunique()}')
print(f'Training samples wrong in more than {TRAIN_WRONG_THRESHOLD} model runs: {n_wrong_gt6_runs}')
print(f'Training samples wrong in at least {TRAIN_WRONG_THRESHOLD} model runs: {n_wrong_ge6_runs}')
print(f'Training samples wrong in more than {TRAIN_WRONG_THRESHOLD} distinct model families: {n_wrong_gt6_families}')

display(train_metrics[['model', 'n_samples', 'accuracy', 'macro_f1', 'roc_auc', 'pr_auc']])
display(train_failure_summary.head(25)[[
    'id', 'source', 'label', 'wrong_train_run_count', 'total_train_model_runs',
    'wrong_train_family_count', 'train_model_families_evaluated', 'title', 'train_models_wrong'
]])

print('Saved:')
print(' - final_artifacts/train_eval_all_model_votes_long.csv')
print(' - final_artifacts/train_eval_wrong_votes_long.csv')
print(' - final_artifacts/train_eval_most_frequent_wrong_samples.csv')
print(' - final_artifacts/train_eval_model_metrics.csv')'''))

    nbformat.write(nb, nb_path)
    print(f"Appended training-set wrong-count diagnostic to {nb_path}")
