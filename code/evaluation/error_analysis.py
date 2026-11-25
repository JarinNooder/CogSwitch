"""
Error analysis for CogSwitch λ=0.7 — dumps all FN and FP with confidence scores.

Run:
    cd /home/nmunir/cogswitch
    python3 error_analysis.py

Output:
    /home/nmunir/cogswitch/error_analysis.txt
"""

import warnings
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import AutoTokenizer, AutoModel
from torch.utils.data import Dataset, DataLoader
warnings.filterwarnings('ignore')

# ── Config ────────────────────────────────────────────────────────────────────
DATA_PATH  = '/home/nmunir/cogswitch/data/final_v2/test.csv'
CKPT_PATH  = '/home/nmunir/cogswitch/checkpoints/best_lam0.7.pt'
OUT_TXT    = '/home/nmunir/cogswitch/error_analysis.txt'
MODEL_NAME = 'xlm-roberta-base'
MAX_LEN    = 128
BATCH_SIZE = 64
DEVICE     = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

print(f"Device: {DEVICE}")

# ── Model ─────────────────────────────────────────────────────────────────────
class CogSwitchModel(nn.Module):
    def __init__(self, model_name=MODEL_NAME, n_persuasion=5):
        super().__init__()
        self.encoder    = AutoModel.from_pretrained(model_name)
        hidden          = self.encoder.config.hidden_size
        self.projector  = nn.Sequential(
            nn.Linear(hidden, hidden), nn.ReLU(), nn.Linear(hidden, 128)
        )
        self.classifier = nn.Linear(hidden, 2)
        self.aux_heads  = nn.ModuleList([nn.Linear(hidden, 2) for _ in range(n_persuasion)])

    def forward(self, input_ids, attention_mask):
        cls    = self.encoder(input_ids=input_ids,
                              attention_mask=attention_mask).last_hidden_state[:, 0, :]
        logits = self.classifier(cls)
        proj   = F.normalize(self.projector(cls), dim=1)
        aux    = [head(cls) for head in self.aux_heads]
        return logits, proj, aux

# ── Dataset ───────────────────────────────────────────────────────────────────
class TextDataset(Dataset):
    def __init__(self, df, tokenizer):
        self.texts  = df['text'].fillna('').astype(str).tolist()
        self.labels = df['label'].astype(int).tolist()
        self.tok    = tokenizer

    def __len__(self): return len(self.texts)

    def __getitem__(self, idx):
        enc = self.tok(self.texts[idx], truncation=True, padding='max_length',
                       max_length=MAX_LEN, return_tensors='pt')
        return {
            'input_ids':      enc['input_ids'].squeeze(0),
            'attention_mask': enc['attention_mask'].squeeze(0),
            'label':          self.labels[idx],
        }

# ── Load ──────────────────────────────────────────────────────────────────────
print("Loading model...")
model = CogSwitchModel().to(DEVICE)
ckpt  = torch.load(CKPT_PATH, map_location=DEVICE)
state = ckpt.get('model_state_dict', ckpt)
if all(k.startswith('module.') for k in state.keys()):
    state = {k[7:]: v for k, v in state.items()}
missing, unexpected = model.load_state_dict(state, strict=True)
assert len(missing) == 0 and len(unexpected) == 0
model.eval()

tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)

print("Loading test data...")
test_df = pd.read_csv(DATA_PATH).dropna(subset=['text', 'label', 'lang_pair'])
test_df['label'] = test_df['label'].astype(int)
test_df['text']  = test_df['text'].fillna('').astype(str)
test_df = test_df.reset_index(drop=True)

# ── Inference ─────────────────────────────────────────────────────────────────
loader = DataLoader(TextDataset(test_df, tokenizer),
                    batch_size=BATCH_SIZE, shuffle=False, num_workers=2)
all_preds, all_probs = [], []
with torch.no_grad():
    for batch in loader:
        logits, _, _ = model(batch['input_ids'].to(DEVICE),
                             batch['attention_mask'].to(DEVICE))
        probs = torch.softmax(logits, dim=1)[:, 1].cpu().numpy()
        preds = (probs >= 0.5).astype(int)
        all_preds.extend(preds)
        all_probs.extend(probs)

test_df['pred']          = all_preds
test_df['p_phishing']    = all_probs

fn_df = test_df[(test_df['label'] == 1) & (test_df['pred'] == 0)].copy()  # missed phishing
fp_df = test_df[(test_df['label'] == 0) & (test_df['pred'] == 1)].copy()  # false alarm

print(f"\nFalse Negatives (missed phishing): {len(fn_df)}")
print(f"False Positives (false alarm):      {len(fp_df)}")

# ── Write report ──────────────────────────────────────────────────────────────
lines = []
lines.append("=" * 80)
lines.append(f"ERROR ANALYSIS — CogSwitch λ=0.7")
lines.append(f"Test set: {len(test_df)} samples")
lines.append(f"False Negatives (phishing predicted as ham): {len(fn_df)}")
lines.append(f"False Positives (ham predicted as phishing): {len(fp_df)}")
lines.append("=" * 80)

lines.append("\n" + "─" * 80)
lines.append("FALSE NEGATIVES (missed phishing)")
lines.append("─" * 80)
for _, row in fn_df.sort_values('p_phishing').iterrows():
    lines.append(f"\n[lang_pair: {row['lang_pair']}]  P(phishing)={row['p_phishing']:.4f}")
    if 'category' in row:
        lines.append(f"[category: {row['category']}]")
    lines.append(f"TEXT: {row['text']}")

lines.append("\n" + "─" * 80)
lines.append("FALSE POSITIVES (ham predicted as phishing)")
lines.append("─" * 80)
for _, row in fp_df.sort_values('p_phishing', ascending=False).iterrows():
    lines.append(f"\n[lang_pair: {row['lang_pair']}]  P(phishing)={row['p_phishing']:.4f}")
    if 'category' in row:
        lines.append(f"[category: {row['category']}]")
    lines.append(f"TEXT: {row['text']}")

lines.append("\n" + "=" * 80)
lines.append("SUMMARY BY LANGUAGE PAIR")
lines.append("─" * 80)
for pair in sorted(test_df['lang_pair'].unique()):
    sub = test_df[test_df['lang_pair'] == pair]
    fn  = ((sub['label'] == 1) & (sub['pred'] == 0)).sum()
    fp  = ((sub['label'] == 0) & (sub['pred'] == 1)).sum()
    lines.append(f"  {pair:<8}  FN={fn}  FP={fp}  (n={len(sub)})")

report = "\n".join(lines)
print(report)

with open(OUT_TXT, 'w', encoding='utf-8') as f:
    f.write(report)
print(f"\nSaved -> {OUT_TXT}")
