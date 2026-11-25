"""
Baseline models for CogSwitch phishing detection.
Models: mBERT, SVM, Logistic Regression, Random Forest, MLP
Run: python3 baselines.py
Output: baselines_results.json, baselines_results.txt
"""

import pandas as pd
import numpy as np
import json, os, time
from sklearn.svm import LinearSVC
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.metrics import f1_score, precision_score, recall_score, accuracy_score
from sklearn.preprocessing import LabelEncoder
import warnings
warnings.filterwarnings('ignore')

DATA_DIR = '/home/nmunir/cogswitch/data/final_v2'
OUT_JSON = '/home/nmunir/cogswitch/baselines_results.json'
OUT_TXT  = '/home/nmunir/cogswitch/baselines_results.txt'

# ── Load data ────────────────────────────────────────────────────────────────
print("Loading data...")
train_df = pd.read_csv(f'{DATA_DIR}/train.csv')
test_df  = pd.read_csv(f'{DATA_DIR}/test.csv')

# Drop rows with missing text or label
train_df = train_df.dropna(subset=['text', 'label'])
test_df  = test_df.dropna(subset=['text', 'label'])

train_df['label'] = train_df['label'].astype(int)
test_df['label']  = test_df['label'].astype(int)

print(f"Train: {len(train_df)} | Test: {len(test_df)}")

y_train = train_df['label'].values
y_test  = test_df['label'].values
lang_pairs = test_df['lang_pair'].values

# ── Metrics helper ───────────────────────────────────────────────────────────
def evaluate(y_true, y_pred, lang_pairs):
    result = {
        'macro_f1':   round(f1_score(y_true, y_pred, average='macro'), 4),
        'pos_f1':     round(f1_score(y_true, y_pred, pos_label=1), 4),
        'accuracy':   round(accuracy_score(y_true, y_pred), 4),
        'precision':  round(precision_score(y_true, y_pred, average='macro'), 4),
        'recall':     round(recall_score(y_true, y_pred, average='macro'), 4),
        'per_lang': {}
    }
    for lp in ['en_hi', 'en_ar', 'en_es', 'en_zh', 'en']:
        mask = lang_pairs == lp
        if mask.sum() > 0:
            result['per_lang'][lp] = round(f1_score(y_true[mask], y_pred[mask], average='macro', zero_division=0), 4)
    return result

results = {}

# ── 1. TF-IDF + Traditional ML ──────────────────────────────────────────────
print("\nFitting TF-IDF...")
from sklearn.feature_extraction.text import TfidfVectorizer
tfidf = TfidfVectorizer(max_features=50000, ngram_range=(1,2), sublinear_tf=True)
X_train_tfidf = tfidf.fit_transform(train_df['text'].fillna(''))
X_test_tfidf  = tfidf.transform(test_df['text'].fillna(''))
print(f"TF-IDF shape: {X_train_tfidf.shape}")

# SVM
print("Training SVM...")
t0 = time.time()
svm = LinearSVC(C=1.0, max_iter=2000)
svm.fit(X_train_tfidf, y_train)
y_pred_svm = svm.predict(X_test_tfidf)
results['SVM (TF-IDF)'] = evaluate(y_test, y_pred_svm, lang_pairs)
results['SVM (TF-IDF)']['time_s'] = round(time.time()-t0, 1)
print(f"  macro-F1: {results['SVM (TF-IDF)']['macro_f1']}  [{results['SVM (TF-IDF)']['time_s']}s]")

# Logistic Regression
print("Training Logistic Regression...")
t0 = time.time()
lr = LogisticRegression(C=1.0, max_iter=1000, solver='lbfgs')
lr.fit(X_train_tfidf, y_train)
y_pred_lr = lr.predict(X_test_tfidf)
results['LR (TF-IDF)'] = evaluate(y_test, y_pred_lr, lang_pairs)
results['LR (TF-IDF)']['time_s'] = round(time.time()-t0, 1)
print(f"  macro-F1: {results['LR (TF-IDF)']['macro_f1']}  [{results['LR (TF-IDF)']['time_s']}s]")

# Random Forest
print("Training Random Forest...")
t0 = time.time()
rf = RandomForestClassifier(n_estimators=200, n_jobs=-1, random_state=42)
rf.fit(X_train_tfidf, y_train)
y_pred_rf = rf.predict(X_test_tfidf)
results['RF (TF-IDF)'] = evaluate(y_test, y_pred_rf, lang_pairs)
results['RF (TF-IDF)']['time_s'] = round(time.time()-t0, 1)
print(f"  macro-F1: {results['RF (TF-IDF)']['macro_f1']}  [{results['RF (TF-IDF)']['time_s']}s]")

# MLP
print("Training MLP...")
t0 = time.time()
mlp = MLPClassifier(hidden_layer_sizes=(512, 128), max_iter=50, random_state=42, early_stopping=True)
mlp.fit(X_train_tfidf, y_train)
y_pred_mlp = mlp.predict(X_test_tfidf)
results['MLP (TF-IDF)'] = evaluate(y_test, y_pred_mlp, lang_pairs)
results['MLP (TF-IDF)']['time_s'] = round(time.time()-t0, 1)
print(f"  macro-F1: {results['MLP (TF-IDF)']['macro_f1']}  [{results['MLP (TF-IDF)']['time_s']}s]")

# ── 2. mBERT ─────────────────────────────────────────────────────────────────
print("\nTraining mBERT...")
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from transformers import AutoTokenizer, AutoModel
from torch.optim import AdamW

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"Using device: {device}")

class SimpleDataset(Dataset):
    def __init__(self, df, tokenizer, max_len=128):
        self.texts  = df['text'].fillna('').tolist()
        self.labels = df['label'].tolist()
        self.tok    = tokenizer
        self.max_len = max_len
    def __len__(self): return len(self.texts)
    def __getitem__(self, idx):
        enc = self.tok(self.texts[idx], truncation=True, padding='max_length',
                       max_length=self.max_len, return_tensors='pt')
        return {
            'input_ids':      enc['input_ids'].squeeze(0),
            'attention_mask': enc['attention_mask'].squeeze(0),
            'label':          torch.tensor(self.labels[idx], dtype=torch.long)
        }

class BertClassifier(nn.Module):
    def __init__(self, model_name):
        super().__init__()
        self.encoder   = AutoModel.from_pretrained(model_name)
        self.classifier = nn.Linear(768, 2)
    def forward(self, input_ids, attention_mask):
        out = self.encoder(input_ids=input_ids, attention_mask=attention_mask)
        cls = out.last_hidden_state[:, 0, :]
        return self.classifier(cls)

mbert_name = 'bert-base-multilingual-cased'
tokenizer  = AutoTokenizer.from_pretrained(mbert_name)
model      = BertClassifier(mbert_name).to(device)
if torch.cuda.device_count() > 1:
    model = nn.DataParallel(model)

train_ds = SimpleDataset(train_df, tokenizer)
test_ds  = SimpleDataset(test_df,  tokenizer)
train_loader = DataLoader(train_ds, batch_size=32, shuffle=True,  num_workers=2)
test_loader  = DataLoader(test_ds,  batch_size=64, shuffle=False, num_workers=2)

optimizer = AdamW(model.parameters(), lr=2e-5)
criterion = nn.CrossEntropyLoss()

t0 = time.time()
# Train 3 epochs
for epoch in range(3):
    model.train()
    total_loss = 0
    for batch in train_loader:
        ids  = batch['input_ids'].to(device)
        mask = batch['attention_mask'].to(device)
        lbls = batch['label'].to(device)
        optimizer.zero_grad()
        logits = model(ids, mask)
        loss   = criterion(logits, lbls)
        loss.backward()
        optimizer.step()
        total_loss += loss.item()
    print(f"  Epoch {epoch+1}/3 — loss: {total_loss/len(train_loader):.4f}")

# Evaluate
model.eval()
preds = []
with torch.no_grad():
    for batch in test_loader:
        ids  = batch['input_ids'].to(device)
        mask = batch['attention_mask'].to(device)
        logits = model(ids, mask)
        preds.extend(logits.argmax(dim=1).cpu().tolist())

y_pred_mbert = np.array(preds)
results['mBERT'] = evaluate(y_test, y_pred_mbert, lang_pairs)
results['mBERT']['time_s'] = round(time.time()-t0, 1)
print(f"  macro-F1: {results['mBERT']['macro_f1']}  [{results['mBERT']['time_s']}s]")

# ── Print summary ─────────────────────────────────────────────────────────────
lines = []
lines.append("=" * 70)
lines.append("BASELINE RESULTS")
lines.append("=" * 70)
lines.append(f"{'Model':<20} {'Macro-F1':>10} {'Pos-F1':>10} {'Acc':>8} {'en_hi':>8} {'en_ar':>8} {'en_es':>8} {'en_zh':>8}")
lines.append("-" * 70)
for name, r in results.items():
    lp = r.get('per_lang', {})
    lines.append(f"{name:<20} {r['macro_f1']:>10.4f} {r['pos_f1']:>10.4f} {r['accuracy']:>8.4f}"
                 f" {lp.get('en_hi','—'):>8} {lp.get('en_ar','—'):>8} {lp.get('en_es','—'):>8} {lp.get('en_zh','—'):>8}")

# Add our best model for comparison
lines.append("-" * 70)
lines.append(f"{'XLM-R+SupCon λ=0.3':<20} {'0.9940':>10} {'0.9926':>10} {'—':>8} {'1.0000':>8} {'0.9967':>8} {'1.0000':>8} {'1.0000':>8}")
lines.append("=" * 70)

summary = "\n".join(lines)
print("\n" + summary)

with open(OUT_TXT, 'w') as f:
    f.write(summary)

with open(OUT_JSON, 'w') as f:
    json.dump(results, f, indent=2)

print(f"\nSaved → {OUT_JSON}")
print(f"Saved → {OUT_TXT}")
