"""
t-SNE: XLM-R + Aux (λ=0.0) vs CogSwitch (λ=0.7) — Fig. 1.
Uses encoder CLS embeddings (768-dim), 1000 test samples, seed=42.
Qualitative visualization only; independently fitted t-SNE coordinates are not scale-comparable.

Run:
    cd /home/nmunir/cogswitch
    python3 tsne_viz.py

Output:
    /home/nmunir/cogswitch/tsne_comparison.png
"""

import os, warnings
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from transformers import AutoTokenizer, AutoModel
from torch.utils.data import Dataset, DataLoader
from sklearn.manifold import TSNE
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
warnings.filterwarnings('ignore')

# ── Config ────────────────────────────────────────────────────────────────────
DATA_PATH   = '/home/nmunir/cogswitch/data/final_v2/test.csv'
CKPT_CE     = '/home/nmunir/cogswitch/checkpoints/best_lam0.0.pt'
CKPT_SC     = '/home/nmunir/cogswitch/checkpoints/best_lam0.7.pt'
OUT_PNG     = '/home/nmunir/cogswitch/tsne_comparison.png'
MODEL_NAME  = 'xlm-roberta-base'
MAX_LEN     = 128
BATCH_SIZE  = 64
N_SAMPLES   = 1000
SEED        = 42
DEVICE      = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

print(f"Device : {DEVICE}")

# ── Model ─────────────────────────────────────────────────────────────────────
class CogSwitchModel(nn.Module):
    def __init__(self, model_name=MODEL_NAME, n_persuasion=5):
        super().__init__()
        self.encoder   = AutoModel.from_pretrained(model_name)
        hidden         = self.encoder.config.hidden_size
        self.projector = nn.Sequential(
            nn.Linear(hidden, hidden), nn.ReLU(), nn.Linear(hidden, 128)
        )
        self.classifier = nn.Linear(hidden, 2)
        self.aux_heads  = nn.ModuleList([nn.Linear(hidden, 2) for _ in range(n_persuasion)])

    def forward(self, input_ids, attention_mask):
        cls  = self.encoder(input_ids=input_ids,
                            attention_mask=attention_mask).last_hidden_state[:, 0, :]
        return cls   # encoder CLS representation (B, 768)

# ── Dataset ───────────────────────────────────────────────────────────────────
class TextDataset(Dataset):
    def __init__(self, df, tokenizer):
        self.texts  = df['text'].fillna('').astype(str).tolist()
        self.labels = df['label'].astype(int).tolist()
        self.pairs  = df['lang_pair'].tolist()
        self.tok    = tokenizer

    def __len__(self): return len(self.texts)

    def __getitem__(self, idx):
        enc = self.tok(self.texts[idx], truncation=True, padding='max_length',
                       max_length=MAX_LEN, return_tensors='pt')
        return {
            'input_ids':      enc['input_ids'].squeeze(0),
            'attention_mask': enc['attention_mask'].squeeze(0),
            'label':          self.labels[idx],
            'lang_pair':      self.pairs[idx],
        }

# ── Extract encoder CLS embeddings ───────────────────────────────────────────
def extract(ckpt_path, df, tokenizer):
    model = CogSwitchModel().to(DEVICE)
    ckpt  = torch.load(ckpt_path, map_location=DEVICE)
    state = ckpt.get('model_state_dict', ckpt)
    if all(k.startswith('module.') for k in state.keys()):
        state = {k[7:]: v for k, v in state.items()}
    missing, unexpected = model.load_state_dict(state, strict=True)
    assert len(missing) == 0,    f"Missing: {missing}"
    assert len(unexpected) == 0, f"Unexpected: {unexpected}"
    model.eval()

    loader = DataLoader(TextDataset(df, tokenizer),
                        batch_size=BATCH_SIZE, shuffle=False, num_workers=2)
    embeddings, labels, pairs = [], [], []
    with torch.no_grad():
        for batch in loader:
            cls = model(batch['input_ids'].to(DEVICE),
                        batch['attention_mask'].to(DEVICE))
            embeddings.append(cls.cpu().numpy())
            labels.extend(batch['label'])
            pairs.extend(batch['lang_pair'])

    return np.vstack(embeddings), np.array(labels), np.array(pairs)

# ── t-SNE + plot ──────────────────────────────────────────────────────────────
def plot_tsne(ax, emb, labels, pairs, title):
    print(f"  Running t-SNE for: {title}")
    tsne_kwargs = dict(n_components=2, perplexity=40, random_state=SEED)
    try:
        coords = TSNE(**tsne_kwargs, max_iter=1000).fit_transform(emb)   # sklearn >=1.4
    except TypeError:
        coords = TSNE(**tsne_kwargs, n_iter=1000).fit_transform(emb)    # sklearn <1.4

    cs_mask   = np.array([p != 'en' for p in pairs])
    mono_mask = ~cs_mask
    COLOR = {0: '#2196F3', 1: '#F44336'}   # blue=ham, red=phishing

    for lbl in [0, 1]:
        for mask, marker in [(cs_mask, 'o'), (mono_mask, 's')]:
            m = (labels == lbl) & mask
            if m.sum() == 0: continue
            ax.scatter(coords[m, 0], coords[m, 1],
                       c=COLOR[lbl], marker=marker,
                       alpha=0.45, s=14, linewidths=0)

    ax.set_title(title, fontsize=12, fontweight='bold')
    ax.set_xticks([]); ax.set_yticks([])
    ax.spines['top'].set_visible(False); ax.spines['right'].set_visible(False)
    ax.spines['bottom'].set_visible(False); ax.spines['left'].set_visible(False)

    # No cross-panel distance comparison: each t-SNE is fitted independently.

# ── Main ──────────────────────────────────────────────────────────────────────
print("Loading data...")
test_df = pd.read_csv(DATA_PATH).dropna(subset=['text', 'label', 'lang_pair'])
test_df['label'] = test_df['label'].astype(int)
if len(test_df) > N_SAMPLES:
    test_df = test_df.sample(N_SAMPLES, random_state=SEED).reset_index(drop=True)
print(f"Samples : {len(test_df)}")
print(test_df['lang_pair'].value_counts())

tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)

print("\nExtracting XLM-R + Aux (λ=0.0) CLS embeddings...")
cls_ce, labels, pairs = extract(CKPT_CE, test_df, tokenizer)

print("Extracting CogSwitch (λ=0.7) CLS embeddings...")
cls_sc, _, _ = extract(CKPT_SC, test_df, tokenizer)

# ── Plot ──────────────────────────────────────────────────────────────────────
fig, axes = plt.subplots(1, 2, figsize=(14, 6))
fig.suptitle(
    't-SNE: Encoder CLS Representation Comparison\n'
    'XLM-R + Aux (λ=0.0) vs CogSwitch (λ=0.7)',
    fontsize=13, fontweight='bold', y=1.01
)

plot_tsne(axes[0], cls_ce, labels, pairs, 'XLM-R + Aux (λ=0.0)\nCLS embeddings')
plot_tsne(axes[1], cls_sc, labels, pairs, 'CogSwitch (λ=0.7)\nCLS embeddings')

# Shared legend
color_handles = [
    Line2D([0],[0], marker='o', color='w', markerfacecolor='#F44336',
           markersize=8, label='Phishing'),
    Line2D([0],[0], marker='o', color='w', markerfacecolor='#2196F3',
           markersize=8, label='Ham'),
]
marker_handles = [
    Line2D([0],[0], marker='o', color='gray', linestyle='None',
           markersize=7, label='Code-switching'),
    Line2D([0],[0], marker='s', color='gray', linestyle='None',
           markersize=7, label='English-source'),
]
fig.legend(handles=color_handles + marker_handles,
           loc='lower center', ncol=4, fontsize=9,
           bbox_to_anchor=(0.5, -0.04))

plt.tight_layout()
plt.savefig(OUT_PNG, dpi=150, bbox_inches='tight')
plt.close()
print(f"\nSaved -> {OUT_PNG}")
print("\nDone.")
