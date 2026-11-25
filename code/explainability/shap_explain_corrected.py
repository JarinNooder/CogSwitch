"""SHAP word-level explanations for CogSwitch (lambda=0.7).
200 fixed-seed test examples. Mean absolute SHAP attribution conditional on
word occurrence (minimum two occurrences). No unsupported automatic mapping
from words to persuasion mechanisms. Saves a muted-color Fig. 5 and JSON.
"""

import os, json, warnings, re
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import AutoTokenizer, AutoModel
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
warnings.filterwarnings('ignore')

# ── Config ────────────────────────────────────────────────────────────────────
DATA_DIR   = '/home/nmunir/cogswitch/data/final_v2'
CKPT_PATH  = '/home/nmunir/cogswitch/checkpoints/best_lam0.7.pt'
OUT_PNG    = '/home/nmunir/cogswitch/shap_top_features.png'
OUT_JSON   = '/home/nmunir/cogswitch/shap_results.json'
MODEL_NAME = 'xlm-roberta-base'
MAX_LEN    = 128
BATCH_SIZE = 32
N_SAMPLES  = 200
SEED       = 42
DEVICE     = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

print(f"Device    : {DEVICE}")
print(f"Checkpoint: {CKPT_PATH}")

# ── Model (matches train_cogswitch.py exactly) ────────────────────────────────
class CogSwitchModel(nn.Module):
    def __init__(self, model_name, n_persuasion=5):
        super().__init__()
        self.encoder    = AutoModel.from_pretrained(model_name)
        hidden          = self.encoder.config.hidden_size
        self.projector  = nn.Sequential(
            nn.Linear(hidden, hidden),
            nn.ReLU(),
            nn.Linear(hidden, 128)
        )
        self.classifier = nn.Linear(hidden, 2)
        self.aux_heads  = nn.ModuleList([nn.Linear(hidden, 2) for _ in range(n_persuasion)])

    def forward(self, input_ids, attention_mask):
        out    = self.encoder(input_ids=input_ids, attention_mask=attention_mask)
        cls    = out.last_hidden_state[:, 0, :]
        logits = self.classifier(cls)
        proj   = F.normalize(self.projector(cls), dim=1)
        aux    = [head(cls) for head in self.aux_heads]
        return logits, proj, aux

# ── Load model ────────────────────────────────────────────────────────────────
print("Loading model...")
model = CogSwitchModel(MODEL_NAME).to(DEVICE)
ckpt  = torch.load(CKPT_PATH, map_location=DEVICE)
state = ckpt.get('model_state_dict', ckpt)
if all(k.startswith('module.') for k in state.keys()):
    state = {k[7:]: v for k, v in state.items()}
missing, unexpected = model.load_state_dict(state, strict=True)
assert len(missing) == 0,    f"Missing keys: {missing}"
assert len(unexpected) == 0, f"Unexpected keys: {unexpected}"
model.eval()
print("Model loaded OK — Missing=[], Unexpected=[]")

tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)

# ── Predict function — returns scalar P(phishing) per sample ─────────────────
def predict_phish_prob(texts):
    texts = list(texts)
    out   = []
    for i in range(0, len(texts), BATCH_SIZE):
        batch = texts[i:i + BATCH_SIZE]
        enc   = tokenizer(batch, truncation=True, padding=True,
                          max_length=MAX_LEN, return_tensors='pt')
        with torch.no_grad():
            logits, _, _ = model(enc['input_ids'].to(DEVICE),
                                 enc['attention_mask'].to(DEVICE))
        probs = torch.softmax(logits, dim=1)[:, 1].cpu().numpy()
        out.append(probs)
    return np.concatenate(out)

# ── Load test data ────────────────────────────────────────────────────────────
print("Loading test data...")
test_df = pd.read_csv(f'{DATA_DIR}/test.csv').dropna(subset=['text', 'label'])
test_df['label'] = test_df['label'].astype(int)
test_df['text']  = test_df['text'].fillna('').astype(str)
test_df = test_df[test_df['text'].str.split().str.len() >= 5].reset_index(drop=True)
print(f"Eligible rows: {len(test_df)}")

if len(test_df) < N_SAMPLES:
    raise RuntimeError(f"Need {N_SAMPLES} eligible rows, found {len(test_df)}")
sample_df = test_df.sample(n=N_SAMPLES, random_state=SEED)
texts     = sample_df['text'].tolist()
print(f"Sampled {len(texts)} instances for SHAP")

# ── SHAP ──────────────────────────────────────────────────────────────────────
import shap

masker    = shap.maskers.Text(tokenizer=r"\W+")   # split on non-word chars
explainer = shap.Explainer(predict_phish_prob, masker)

_punct_re = re.compile(r"[^\w؀-ۿऀ-ॿ一-鿿]+", re.UNICODE)

def clean_token(w):
    """Strip leading/trailing punctuation; lowercase."""
    return _punct_re.sub('', w).strip().lower()

word_importance = {}  # token -> list of |SHAP| for observed occurrences
failures = []
for i, text in enumerate(texts):
    try:
        sv = explainer([text])
        vals = np.asarray(sv.values[0]).reshape(-1)
        words = sv.data[0]
        if len(vals) != len(words):
            raise ValueError(f"Token/attribution length mismatch: {len(words)} vs {len(vals)}")
        if not np.all(np.isfinite(vals)):
            raise ValueError("Nonfinite SHAP values")
        for word, val in zip(words, vals):
            word = clean_token(str(word))
            if len(word) < 2 or word.isdigit() or re.fullmatch(r'[0-9a-f]{20,}', word):
                continue
            word_importance.setdefault(word, []).append(abs(float(val)))
        if (i + 1) % 25 == 0:
            print(f"  {i + 1}/{len(texts)} done")
    except Exception as e:
        failures.append((i, str(e)))
        print(f"  SHAP failed on sample {i}: {type(e).__name__}: {e}")

if failures:
    raise RuntimeError(
        f"SHAP failed on {len(failures)}/{len(texts)} samples. "
        "Fix the errors before using this figure; no partial results were saved."
    )
print(f"Completed all {len(texts)} SHAP explanations")

word_scores = {
    w: float(np.mean(v))
    for w, v in word_importance.items()
    if len(v) >= 2
}
top_words = sorted(word_scores.items(), key=lambda x: x[1], reverse=True)[:20]
if not top_words:
    raise RuntimeError("No tokens with at least two occurrences; cannot plot")
print(f"Top 10 tokens: {[w for w, _ in top_words[:10]]}")

# ── Save JSON ─────────────────────────────────────────────────────────────────
save = {
    'n_samples': len(texts),
    'score_definition': 'Mean absolute SHAP attribution conditional on token occurrence',
    'min_token_occurrences': 2,
    'top_words': [
        {'token': w, 'mean_abs_shap': v, 'occurrences': len(word_importance[w])}
        for w, v in top_words
    ],
}
with open(OUT_JSON, 'w', encoding='utf-8') as f:
    json.dump(save, f, indent=2, ensure_ascii=False)
print(f"\nSaved -> {OUT_JSON}")

# ── Muted, colorblind-friendly bar chart ──────────────────────────────────────
words20 = [w for w, _ in top_words]
scores20 = [v for _, v in top_words]
fig, ax = plt.subplots(figsize=(10, 7))
ax.barh(range(len(words20)), scores20,
        color='#8CAFC1', edgecolor='#F6F8FA', linewidth=0.6, height=0.72)
ax.set_yticks(range(len(words20)))
ax.set_yticklabels(words20, fontsize=10)
ax.invert_yaxis()
ax.set_xlabel('Mean absolute SHAP value (among token occurrences)', fontsize=10)
ax.set_title('Top-20 Tokens by SHAP Attribution\n'
             'CogSwitch (λ=0.7) | 200 test messages', fontsize=12, fontweight='bold')
ax.grid(axis='x', alpha=0.18, linewidth=0.7)
ax.set_axisbelow(True)
ax.spines['top'].set_visible(False)
ax.spines['right'].set_visible(False)
max_score = max(scores20)
ax.set_xlim(0, max_score * 1.20)
for i, score in enumerate(scores20):
    ax.text(score + max_score * 0.012, i, f'{score:.4f}', va='center', fontsize=8,
            color='#4A5560')
plt.tight_layout()
plt.savefig(OUT_PNG, dpi=180, bbox_inches='tight')
plt.close()
print(f'Saved -> {OUT_PNG}')
print('\nDone.')
