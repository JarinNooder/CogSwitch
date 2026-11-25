"""
Fairness evaluation for CogSwitch phishing detection.
Groups: code-switched language pairs (en_hi, en_ar, en_es, en_zh).
Models: CogSwitch checkpoints for lambda = 0.0, 0.3, 0.5, 0.7, 0.9.

Run after the corrected lambda sweep has finished:
    python3 fairness_eval_v2_fixed.py

Outputs (in /home/nmunir/cogswitch/):
    fairness_results_v2.json
    fairness_results_v2.txt
    fairness_fpr_fnr.png

The script evaluates saved checkpoints; it does not train or modify them.
The lambda=0.7 checkpoint must be the intended full model (aux weight 0.1).
"""

import hashlib
import json
import os
import time
import warnings

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset
from transformers import AutoModel, AutoTokenizer

warnings.filterwarnings('ignore')

DATA_DIR = '/home/nmunir/cogswitch/data/final_v2'
CKPT_DIR = '/home/nmunir/cogswitch/checkpoints'
OUT_JSON = '/home/nmunir/cogswitch/fairness_results_v2.json'
OUT_TXT = '/home/nmunir/cogswitch/fairness_results_v2.txt'
OUT_CHART = '/home/nmunir/cogswitch/fairness_fpr_fnr.png'

MODEL_NAME = 'xlm-roberta-base'
BATCH_SIZE = 64
MAX_LEN = 128
DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
N_BOOTSTRAP = 10000
SEED = 42
CS_GROUPS = ['en_hi', 'en_ar', 'en_es', 'en_zh']
LAM_VALUES = ['0.0', '0.3', '0.5', '0.7', '0.9']

print(f'Device: {DEVICE}')


# ── Model definition (must match the saved training architecture) ─────────────
class CogSwitchModel(nn.Module):
    def __init__(self, model_name, n_persuasion=5):
        super().__init__()
        self.encoder = AutoModel.from_pretrained(model_name)
        hidden = self.encoder.config.hidden_size
        self.projector = nn.Sequential(
            nn.Linear(hidden, hidden),
            nn.ReLU(),
            nn.Linear(hidden, 128),
        )
        self.classifier = nn.Linear(hidden, 2)
        self.aux_heads = nn.ModuleList(
            [nn.Linear(hidden, 2) for _ in range(n_persuasion)]
        )

    def forward(self, input_ids, attention_mask):
        out = self.encoder(input_ids=input_ids, attention_mask=attention_mask)
        cls = out.last_hidden_state[:, 0, :]
        logits = self.classifier(cls)
        proj = F.normalize(self.projector(cls), dim=1)
        aux = [head(cls) for head in self.aux_heads]
        return logits, proj, aux


# ── Dataset ───────────────────────────────────────────────────────────────────
class PhishingDataset(Dataset):
    def __init__(self, df, tokenizer, max_len=128):
        self.texts = df['text'].fillna('').tolist()
        self.labels = df['label'].tolist()
        self.tok = tokenizer
        self.max_len = max_len

    def __len__(self):
        return len(self.texts)

    def __getitem__(self, idx):
        enc = self.tok(
            self.texts[idx],
            truncation=True,
            padding='max_length',
            max_length=self.max_len,
            return_tensors='pt',
        )
        return {
            'input_ids': enc['input_ids'].squeeze(0),
            'attention_mask': enc['attention_mask'].squeeze(0),
            'label': torch.tensor(self.labels[idx], dtype=torch.long),
        }


def sha256_file(path):
    """Fingerprint the exact checkpoint bytes used for reproducibility."""
    digest = hashlib.sha256()
    with open(path, 'rb') as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b''):
            digest.update(chunk)
    return digest.hexdigest()


# ── Check inputs before inference; do not mix missing and old checkpoints ────
test_path = os.path.join(DATA_DIR, 'test.csv')
if not os.path.isfile(test_path):
    raise FileNotFoundError(f'Test data not found: {test_path}')

ckpt_paths = {
    lam: os.path.join(CKPT_DIR, f'best_lam{lam}.pt')
    for lam in LAM_VALUES
}
missing = [path for path in ckpt_paths.values() if not os.path.isfile(path)]
if missing:
    raise FileNotFoundError(
        'All five corrected lambda-sweep checkpoints are required. '
        'Missing:\n' + '\n'.join(missing)
    )

print('All five checkpoint files found.')
print('IMPORTANT: Verify these are the newly trained corrected-SupCon checkpoints.')
print('IMPORTANT: best_lam0.7.pt must be the full model (aux weight = 0.1).')

print('Loading test data...')
test_df = pd.read_csv(test_path)
required_cols = {'text', 'label', 'lang_pair'}
if not required_cols.issubset(test_df.columns):
    raise ValueError(f'Missing columns in test.csv: {sorted(required_cols - set(test_df.columns))}')

test_df = test_df.dropna(subset=['text', 'label', 'lang_pair']).copy()
test_df['label'] = test_df['label'].astype(int)
if not test_df['label'].isin([0, 1]).all():
    raise ValueError('Expected binary labels 0 and 1.')
missing_groups = sorted(set(CS_GROUPS) - set(test_df['lang_pair'].unique()))
if missing_groups:
    raise ValueError(f'Missing code-switched groups: {missing_groups}')

print(f'Test set: {len(test_df)} rows')
print(test_df['lang_pair'].value_counts().to_string())

y_true = test_df['label'].to_numpy()
lang_pairs = test_df['lang_pair'].to_numpy()
tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
test_ds = PhishingDataset(test_df, tokenizer, MAX_LEN)
test_loader = DataLoader(test_ds, batch_size=BATCH_SIZE, shuffle=False, num_workers=2)


# ── Inference ─────────────────────────────────────────────────────────────────
def get_predictions(checkpoint_path):
    model = CogSwitchModel(MODEL_NAME).to(DEVICE)
    # Load only trusted, locally produced checkpoints.
    ckpt = torch.load(checkpoint_path, map_location=DEVICE, weights_only=True)
    state = ckpt.get('model_state_dict', ckpt)
    if all(k.startswith('module.') for k in state):
        state = {k[7:]: v for k, v in state.items()}
    model.load_state_dict(state, strict=True)
    model.eval()
    preds = []
    with torch.no_grad():
        for batch in test_loader:
            ids = batch['input_ids'].to(DEVICE)
            mask = batch['attention_mask'].to(DEVICE)
            logits, _, _ = model(ids, mask)
            preds.extend(logits.argmax(dim=1).cpu().tolist())
    del model
    if DEVICE.type == 'cuda':
        torch.cuda.empty_cache()
    return np.asarray(preds)


# ── Per-group TPR/FPR/FNR ─────────────────────────────────────────────────────
def per_group_rates(y_true, y_pred, lang_pairs, groups):
    metrics = {}
    for g in groups:
        mask = lang_pairs == g
        yt, yp = y_true[mask], y_pred[mask]
        tp = int(((yp == 1) & (yt == 1)).sum())
        fn = int(((yp == 0) & (yt == 1)).sum())
        fp = int(((yp == 1) & (yt == 0)).sum())
        tn = int(((yp == 0) & (yt == 0)).sum())
        if tp + fn == 0 or fp + tn == 0:
            raise ValueError(f'Group {g} lacks positive or negative examples.')
        metrics[g] = {
            'n': int(mask.sum()),
            'tpr': round(tp / (tp + fn), 4),
            'fpr': round(fp / (fp + tn), 4),
            'fnr': round(fn / (fn + tp), 4),
            'tp': tp, 'fn': fn, 'fp': fp, 'tn': tn,
        }
    return metrics


# ── Bootstrap CI for EOD (same pooled resampling as original script) ─────────
def bootstrap_eod(y_true, y_pred, lang_pairs, groups, n_boot=10000, seed=42):
    """EOD = max minus min group TPR or FPR; CS groups only."""
    rng = np.random.default_rng(seed)
    cs_mask = np.isin(lang_pairs, groups)
    yt_cs, yp_cs, lp_cs = (
        y_true[cs_mask], y_pred[cs_mask], lang_pairs[cs_mask]
    )

    def eod_tpr_fpr(yt, yp, lp):
        tprs, fprs = [], []
        for g in groups:
            m = lp == g
            tp = int(((yp[m] == 1) & (yt[m] == 1)).sum())
            fn = int(((yp[m] == 0) & (yt[m] == 1)).sum())
            fp = int(((yp[m] == 1) & (yt[m] == 0)).sum())
            tn = int(((yp[m] == 0) & (yt[m] == 0)).sum())
            # An empty class denominator in a bootstrap draw is invalid,
            # rather than silently assigning it a zero rate.
            if tp + fn == 0 or fp + tn == 0:
                return None
            tprs.append(tp / (tp + fn))
            fprs.append(fp / (fp + tn))
        return max(tprs) - min(tprs), max(fprs) - min(fprs)

    point = eod_tpr_fpr(yt_cs, yp_cs, lp_cs)
    if point is None:
        raise ValueError('Cannot calculate EOD: a language group lacks a class.')

    n = len(yt_cs)
    boot_tprs, boot_fprs = [], []
    for _ in range(n_boot):
        idx = rng.integers(0, n, size=n)
        result = eod_tpr_fpr(yt_cs[idx], yp_cs[idx], lp_cs[idx])
        if result is not None:
            boot_tprs.append(result[0])
            boot_fprs.append(result[1])
    if not boot_tprs:
        raise ValueError('No valid bootstrap samples.')
    if len(boot_tprs) != n_boot:
        print(f'Warning: {n_boot - len(boot_tprs)} invalid bootstrap samples excluded.')

    return {
        'eod_tpr': round(point[0], 4),
        'eod_tpr_ci': [round(float(x), 4) for x in np.percentile(boot_tprs, [2.5, 97.5])],
        'eod_fpr': round(point[1], 4),
        'eod_fpr_ci': [round(float(x), 4) for x in np.percentile(boot_fprs, [2.5, 97.5])],
        'valid_bootstrap_samples': len(boot_tprs),
    }


# ── Evaluate every lambda ─────────────────────────────────────────────────────
all_results = {}
for lam in LAM_VALUES:
    ckpt_path = ckpt_paths[lam]
    label = f'λ={lam}'
    print(f'\nRunning {label}...')
    checkpoint_hash = sha256_file(ckpt_path)
    t0 = time.time()
    y_pred = get_predictions(ckpt_path)
    elapsed = round(time.time() - t0, 1)
    if len(y_pred) != len(y_true):
        raise RuntimeError(f'Prediction count mismatch for {label}.')

    pg = per_group_rates(y_true, y_pred, lang_pairs, CS_GROUPS)
    eod = bootstrap_eod(y_true, y_pred, lang_pairs, CS_GROUPS,
                        n_boot=N_BOOTSTRAP, seed=SEED)
    all_results[label] = {
        'checkpoint_path': ckpt_path,
        'checkpoint_sha256': checkpoint_hash,
        'per_group': pg,
        'eod_tpr': eod['eod_tpr'],
        'eod_tpr_ci': eod['eod_tpr_ci'],
        'eod_fpr': eod['eod_fpr'],
        'eod_fpr_ci': eod['eod_fpr_ci'],
        'valid_bootstrap_samples': eod['valid_bootstrap_samples'],
        'inference_time_s': elapsed,
    }
    print(f'  Inference done in {elapsed}s')
    print(f"  EOD-TPR: {eod['eod_tpr']:.4f}  95% CI {eod['eod_tpr_ci']}")
    print(f"  EOD-FPR: {eod['eod_fpr']:.4f}  95% CI {eod['eod_fpr_ci']}")


# ── Save JSON ─────────────────────────────────────────────────────────────────
with open(OUT_JSON, 'w') as f:
    json.dump(all_results, f, indent=2)
print(f'\nSaved → {OUT_JSON}')


# ── Text summary (Tables IX and X) ────────────────────────────────────────────
lines = [
    '=' * 90,
    'FAIRNESS EVALUATION — CogSwitch (code-switched language-pair groups)',
    'English-source (en) excluded from EOD calculation.',
    '=' * 90,
    f'\n{"Model":<22} {"EOD-TPR":>10} {"95% CI (TPR)":>20} {"EOD-FPR":>10} {"95% CI (FPR)":>20}',
    '-' * 90,
]
for label, res in all_results.items():
    ci_tpr = f"[{res['eod_tpr_ci'][0]:.4f}, {res['eod_tpr_ci'][1]:.4f}]"
    ci_fpr = f"[{res['eod_fpr_ci'][0]:.4f}, {res['eod_fpr_ci'][1]:.4f}]"
    lines.append(f"{label:<22} {res['eod_tpr']:>10.4f} {ci_tpr:>20} {res['eod_fpr']:>10.4f} {ci_fpr:>20}")

lines.extend([
    '\n' + '=' * 90,
    'Per language-pair FPR and FNR — CogSwitch (λ=0.7)',
    '=' * 90,
    f'{"Pair":<10} {"N":>6} {"FPR":>8} {"FNR":>8}',
    '-' * 40,
])
best_label = 'λ=0.7'
for g in CS_GROUPS:
    m = all_results[best_label]['per_group'][g]
    lines.append(f"{g:<10} {m['n']:>6} {m['fpr']:>8.4f} {m['fnr']:>8.4f}")

lines.extend([
    '\nNotes:',
    '  EOD-TPR = max(TPR) - min(TPR) across CS language groups.',
    '  EOD-FPR = max(FPR) - min(FPR) across CS language groups.',
    f'  CIs: 95% pooled bootstrap, {N_BOOTSTRAP:,} resamples, seed={SEED}.',
    '  Figure 3 and Table X: four code-switched language pairs only.',
    '  Group labels describe language pairs, not verified nationalities.',
    '  Verify λ=0.7 is the intended full-model checkpoint (aux weight 0.1).',
    '\nCheckpoint SHA-256 fingerprints:',
])
for label, res in all_results.items():
    lines.append(f"  {label}: {res['checkpoint_sha256']}  ({res['checkpoint_path']})")

summary = '\n'.join(lines)
print('\n' + summary)
with open(OUT_TXT, 'w') as f:
    f.write(summary)
print(f'Saved → {OUT_TXT}')


# ── Figure 3: original four-pair layout and styling ──────────────────────────
pg = all_results[best_label]['per_group']
grps = [g for g in CS_GROUPS if g in pg]
fprs = [pg[g]['fpr'] for g in grps]
fnrs = [pg[g]['fnr'] for g in grps]

x = np.arange(len(grps))
width = 0.35
fig, ax = plt.subplots(figsize=(7, 4))
bars_fpr = ax.bar(x - width / 2, fprs, width, label='FPR', color='#4472C4')
bars_fnr = ax.bar(x + width / 2, fnrs, width, label='FNR', color='#ED7D31')

ax.set_xlabel('Language Pair', fontsize=11)
ax.set_ylabel('Rate', fontsize=11)
ax.set_title('False Positive Rate & False Negative Rate per Language Pair (λ=0.7)', fontsize=10)
ax.set_xticks(x)
ax.set_xticklabels(grps)
ax.legend()
ax.set_ylim(0, max(max(fprs), max(fnrs)) * 1.4 + 0.005)

for bars in (bars_fpr, bars_fnr):
    for bar in bars:
        h = bar.get_height()
        if h > 0:
            ax.annotate(
                f'{h:.4f}',
                xy=(bar.get_x() + bar.get_width() / 2, h),
                xytext=(0, 3),
                textcoords='offset points',
                ha='center', va='bottom', fontsize=8,
            )

plt.tight_layout()
plt.savefig(OUT_CHART, dpi=150, bbox_inches='tight')
plt.close()
print(f'Saved → {OUT_CHART}')
print('\nDone.')
