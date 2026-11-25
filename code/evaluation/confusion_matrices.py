"""
Confusion matrices per language pair — CogSwitch λ=0.7 (Fig. 2 replacement).
2×3 grid layout to address reviewer readability comment.

Run:
    cd /home/nmunir/cogswitch
    python3 confusion_matrices.py

Output:
    /home/nmunir/cogswitch/confusion_matrices.png
"""

import warnings
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import AutoTokenizer, AutoModel
from torch.utils.data import Dataset, DataLoader
from sklearn.metrics import confusion_matrix
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
warnings.filterwarnings('ignore')

# ── Config ────────────────────────────────────────────────────────────────────
DATA_PATH  = '/home/nmunir/cogswitch/data/final_v2/test.csv'
CKPT_PATH  = '/home/nmunir/cogswitch/checkpoints/best_lam0.7.pt'
OUT_PNG    = '/home/nmunir/cogswitch/confusion_matrices.png'
MODEL_NAME = 'xlm-roberta-base'
MAX_LEN    = 128
BATCH_SIZE = 64
DEVICE     = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

PAIRS_ORDER = ['en', 'en_hi', 'en_ar', 'en_es', 'en_zh']
PAIR_LABELS = {
    'en':    'English\n(Monolingual)',
    'en_hi': 'English–Hindi\n(CS)',
    'en_ar': 'English–Arabic\n(CS)',
    'en_es': 'English–Spanish\n(CS)',
    'en_zh': 'English–Chinese\n(CS)',
}

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

# ── Load model ────────────────────────────────────────────────────────────────
print("Loading model...")
model = CogSwitchModel().to(DEVICE)
ckpt  = torch.load(CKPT_PATH, map_location=DEVICE)
state = ckpt.get('model_state_dict', ckpt)
if all(k.startswith('module.') for k in state.keys()):
    state = {k[7:]: v for k, v in state.items()}
missing, unexpected = model.load_state_dict(state, strict=True)
assert len(missing) == 0 and len(unexpected) == 0
model.eval()
print("Model loaded OK")

tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)

# ── Load data ─────────────────────────────────────────────────────────────────
print("Loading test data...")
test_df = pd.read_csv(DATA_PATH).dropna(subset=['text', 'label', 'lang_pair'])
test_df['label'] = test_df['label'].astype(int)
print(test_df['lang_pair'].value_counts())

# ── Run inference ─────────────────────────────────────────────────────────────
loader = DataLoader(TextDataset(test_df, tokenizer),
                    batch_size=BATCH_SIZE, shuffle=False, num_workers=2)
all_preds = []
with torch.no_grad():
    for batch in loader:
        logits, _, _ = model(batch['input_ids'].to(DEVICE),
                             batch['attention_mask'].to(DEVICE))
        preds = torch.argmax(logits, dim=1).cpu().numpy()
        all_preds.extend(preds)

test_df['pred'] = all_preds

# ── Plot 2×3 grid ─────────────────────────────────────────────────────────────
fig, axes = plt.subplots(2, 3, figsize=(15, 9))
axes_flat = axes.flatten()

# hide the 6th panel (5 pairs only)
axes_flat[5].set_visible(False)

def macro_f1_from_cm(cm):
    """Compute macro-F1 from a 2×2 confusion matrix (rows=true, cols=pred)."""
    f1s = []
    for cls in range(2):
        tp = cm[cls, cls]
        fp = cm[:, cls].sum() - tp
        fn = cm[cls, :].sum() - tp
        denom = 2 * tp + fp + fn
        f1s.append((2 * tp / denom) if denom > 0 else 0.0)
    return np.mean(f1s)

for idx, pair in enumerate(PAIRS_ORDER):
    ax  = axes_flat[idx]
    sub = test_df[test_df['lang_pair'] == pair]
    if len(sub) == 0:
        ax.set_visible(False)
        continue

    cm = confusion_matrix(sub['label'], sub['pred'], labels=[0, 1])
    # normalise by row (true class) for background shading
    cm_norm = cm.astype(float) / cm.sum(axis=1, keepdims=True)

    # draw heatmap manually
    im = ax.imshow(cm_norm, cmap='Blues', vmin=0, vmax=1, aspect='auto')

    # annotate with raw counts and per-cell %
    for r in range(2):
        for c in range(2):
            val   = cm[r, c]
            shade = cm_norm[r, c]
            color = 'white' if shade > 0.6 else 'black'
            ax.text(c, r, f"{val}\n({100*shade:.1f}%)", ha='center', va='center',
                    fontsize=13, fontweight='bold', color=color)

    ax.set_xticks([0, 1])
    ax.set_yticks([0, 1])
    ax.set_xticklabels(['Ham\n(Predicted)', 'Phishing\n(Predicted)'], fontsize=10)
    ax.set_yticklabels(['Ham\n(True)', 'Phishing\n(True)'], fontsize=10)
    ax.tick_params(length=0)

    n    = len(sub)
    # TN, FP, FN, TP
    tn, fp, fn, tp = cm[0,0], cm[0,1], cm[1,0], cm[1,1]
    macro_f1 = macro_f1_from_cm(cm)

    print(f"\n{pair}: n={n}  TN={tn}  FP={fp}  FN={fn}  TP={tp}  macro-F1={macro_f1:.4f}")

    ax.set_title(f"{PAIR_LABELS[pair]}\n(n={n},  Macro-F₁={macro_f1:.4f})",
                 fontsize=11, fontweight='bold', pad=8)

fig.suptitle(
    'Confusion Matrices per Language Pair\nCogSwitch (λ=0.7)',
    fontsize=14, fontweight='bold', y=1.01
)
plt.tight_layout()
plt.savefig(OUT_PNG, dpi=150, bbox_inches='tight')
plt.close()
print(f"\nSaved -> {OUT_PNG}")
print("Done.")
