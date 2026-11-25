"""
Auxiliary Heads Ablation — Table XIII equivalent
Two runs controlled by AUX_WEIGHT argument:
    python3 ablation_aux.py --aux_wt 0.0   # SupCon only, no persuasion heads
    python3 ablation_aux.py --aux_wt 0.1   # full model (with aux heads)

Both use λ=0.7, data/final_v2.

Run:
    cd /home/nmunir/cogswitch
    source venv/bin/activate
    python3 -u ablation_aux.py --aux_wt 0.0 2>&1 | tee logs/ablation_noaux.log
    python3 -u ablation_aux.py --aux_wt 0.1 2>&1 | tee logs/ablation_withaux.log
"""

import os, argparse, json, random
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from transformers import AutoTokenizer, AutoModel
from sklearn.metrics import f1_score

# ── Args ──────────────────────────────────────────────────────────────────────
parser = argparse.ArgumentParser()
parser.add_argument('--aux_wt', type=float, required=True,
                    help='Auxiliary head loss weight (0.0 = disabled, 0.1 = full)')
args = parser.parse_args()

# ── Fixed hyperparams ─────────────────────────────────────────────────────────
LAM        = 0.7
AUX_WEIGHT = args.aux_wt
LR         = 2e-5
EPOCHS     = 10
BATCH_SIZE = 32
MAX_LEN    = 128
TEMP       = 0.1
SEED       = 42
MODEL_NAME = 'xlm-roberta-base'
DATA_DIR   = 'data/final_v2'
OUT_DIR    = 'checkpoints'

tag = 'noaux' if AUX_WEIGHT == 0.0 else f'aux{str(AUX_WEIGHT).replace(".", "")}'
CKPT_PATH    = os.path.join(OUT_DIR, f'best_lam0.7_{tag}.pt')
RESULTS_PATH = os.path.join(OUT_DIR, f'results_lam0.7_{tag}.json')

random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)
torch.cuda.manual_seed_all(SEED)

os.makedirs(OUT_DIR, exist_ok=True)
os.makedirs('logs', exist_ok=True)

DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
N_GPU  = torch.cuda.device_count()
print(f"Using {N_GPU} GPU(s): {[torch.cuda.get_device_name(i) for i in range(N_GPU)]}")
print(f"ABLATION: λ={LAM}  aux_weight={AUX_WEIGHT}  data={DATA_DIR}")

# Class weights (same as train_cogswitch.py — computed from training set label distribution)
CLASS_WEIGHTS = torch.tensor([0.77748957, 1.40093477], dtype=torch.float)

PERSUASION_COLS = [
    'persuasion_authority', 'persuasion_social_proof',
    'persuasion_liking', 'persuasion_commitment', 'persuasion_distraction'
]
CATEGORIES = ['cs_phishing', 'cs_ham', 'mono_phishing', 'mono_ham']
LANG_PAIRS = ['en_hi', 'en_ar', 'en_es', 'en_zh']

# ── Dataset ───────────────────────────────────────────────────────────────────
class CogSwitchDataset(Dataset):
    def __init__(self, path, tokenizer, max_len):
        df = pd.read_csv(path)
        self.texts      = df['text'].astype(str).tolist()
        self.labels     = df['label'].tolist()
        self.categories = df['category'].tolist()
        self.lang_pairs = df['lang_pair'].tolist()
        self.persuasion = []
        for _, row in df.iterrows():
            p = []
            for col in PERSUASION_COLS:
                v = row.get(col, float('nan'))
                p.append(-1 if pd.isna(v) else int(v))
            self.persuasion.append(p)
        self.tokenizer = tokenizer
        self.max_len   = max_len

    def __len__(self):
        return len(self.texts)

    def __getitem__(self, idx):
        enc = self.tokenizer(
            self.texts[idx],
            max_length=self.max_len,
            padding='max_length',
            truncation=True,
            return_tensors='pt'
        )
        return {
            'input_ids':      enc['input_ids'].squeeze(0),
            'attention_mask': enc['attention_mask'].squeeze(0),
            'label':          torch.tensor(self.labels[idx], dtype=torch.long),
            'persuasion':     torch.tensor(self.persuasion[idx], dtype=torch.long),
            'category':       self.categories[idx],
            'lang_pair':      self.lang_pairs[idx],
        }

# ── SupCon Loss ───────────────────────────────────────────────────────────────
class SupConLoss(nn.Module):
    def __init__(self, temperature=0.1):
        super().__init__()
        self.temperature = temperature

    def forward(self, features, labels):
        device = features.device
        B = features.shape[0]
        if B < 2:
            return torch.tensor(0.0, device=device)
        sim      = torch.matmul(features, features.T) / self.temperature
        mask_self = torch.eye(B, dtype=torch.bool, device=device)
        sim      = sim.masked_fill(mask_self, float('-inf'))
        labels   = labels.view(-1, 1)
        mask_pos = (labels == labels.T) & ~mask_self
        if mask_pos.sum() == 0:
            return torch.tensor(0.0, device=device)
        log_prob = F.log_softmax(sim, dim=1)
        n_pos    = mask_pos.sum(dim=1).float()
        has_pos  = n_pos > 0
        positive_log_prob = log_prob.masked_fill(~mask_pos, 0.0)
        loss = -(positive_log_prob.sum(dim=1)[has_pos] / n_pos[has_pos]).mean()
        return loss

# ── Model ─────────────────────────────────────────────────────────────────────
class CogSwitchModel(nn.Module):
    def __init__(self, model_name, n_persuasion=5):
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

# ── Metrics ───────────────────────────────────────────────────────────────────
def compute_metrics(all_labels, all_preds, all_cats, all_langs):
    results = {}
    results['binary_f1']     = f1_score(all_labels, all_preds, average='macro')
    results['binary_f1_pos'] = f1_score(all_labels, all_preds, pos_label=1, average='binary')
    for prefix in ['cs', 'mono']:
        idx = [i for i, c in enumerate(all_cats) if c.startswith(prefix)]
        if not idx: continue
        y_t = [all_labels[i] for i in idx]
        y_p = [all_preds[i]  for i in idx]
        results[f'f1_{prefix}'] = f1_score(y_t, y_p, average='macro', zero_division=0)
    for cat in CATEGORIES:
        idx = [i for i, c in enumerate(all_cats) if c == cat]
        if not idx: continue
        correct = sum(all_labels[i] == all_preds[i] for i in idx)
        results[f'acc_{cat}'] = correct / len(idx)
    for lp in LANG_PAIRS:
        idx = [i for i, l in enumerate(all_langs) if l == lp]
        if not idx: continue
        y_t = [all_labels[i] for i in idx]
        y_p = [all_preds[i]  for i in idx]
        results[f'f1_{lp}'] = f1_score(y_t, y_p, average='macro', zero_division=0)
    return results

# ── Epoch ─────────────────────────────────────────────────────────────────────
def run_epoch(model, loader, optimizer, supcon_loss_fn, train=True):
    model.train() if train else model.eval()
    total_loss = 0.0
    all_labels, all_preds, all_cats, all_langs = [], [], [], []
    ctx = torch.enable_grad() if train else torch.no_grad()
    with ctx:
        for batch in loader:
            input_ids = batch['input_ids'].to(DEVICE)
            attn_mask = batch['attention_mask'].to(DEVICE)
            labels    = batch['label'].to(DEVICE)
            persuasion = batch['persuasion'].to(DEVICE)   # (B, 5)

            logits, proj, aux = model(input_ids, attn_mask)

            ce_loss = F.cross_entropy(logits, labels, weight=CLASS_WEIGHTS.to(DEVICE))
            sc_loss = supcon_loss_fn(proj, labels)
            if not torch.isfinite(sc_loss):
                raise RuntimeError(f"SupCon loss is not finite: {sc_loss.item()}")

            # Auxiliary head loss (ignored when AUX_WEIGHT=0.0)
            aux_loss = torch.tensor(0.0, device=DEVICE)
            if AUX_WEIGHT > 0.0:
                for k, head_logits in enumerate(aux):
                    mask = persuasion[:, k] >= 0
                    if mask.sum() > 0:
                        aux_loss = aux_loss + F.cross_entropy(
                            head_logits[mask], persuasion[:, k][mask]
                        )
                aux_loss = aux_loss / len(aux)

            loss = (1 - LAM) * ce_loss + LAM * sc_loss + AUX_WEIGHT * aux_loss
            loss = loss.mean() if loss.dim() > 0 else loss

            if train:
                optimizer.zero_grad()
                loss.backward()
                nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                optimizer.step()

            total_loss += loss.item()
            preds = logits.argmax(dim=1).cpu().tolist()
            all_preds  += preds
            all_labels += labels.cpu().tolist()
            all_cats   += batch['category']
            all_langs  += batch['lang_pair']

    metrics = compute_metrics(all_labels, all_preds, all_cats, all_langs)
    metrics['loss'] = total_loss / len(loader)
    return metrics

# ── Main ──────────────────────────────────────────────────────────────────────
def main():
    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
    print("Loading datasets...")
    train_ds = CogSwitchDataset(f'{DATA_DIR}/train.csv', tokenizer, MAX_LEN)
    val_ds   = CogSwitchDataset(f'{DATA_DIR}/val.csv',   tokenizer, MAX_LEN)
    test_ds  = CogSwitchDataset(f'{DATA_DIR}/test.csv',  tokenizer, MAX_LEN)
    train_loader = DataLoader(train_ds, batch_size=BATCH_SIZE, shuffle=True,
                              num_workers=4, pin_memory=True)
    val_loader   = DataLoader(val_ds,   batch_size=BATCH_SIZE, shuffle=False,
                              num_workers=4, pin_memory=True)
    test_loader  = DataLoader(test_ds,  batch_size=BATCH_SIZE, shuffle=False,
                              num_workers=4, pin_memory=True)

    print("Loading model...")
    model = CogSwitchModel(MODEL_NAME)
    if N_GPU > 1:
        model = nn.DataParallel(model)
    model = model.to(DEVICE)

    optimizer   = torch.optim.AdamW(model.parameters(), lr=LR, weight_decay=0.01)
    supcon_loss = SupConLoss(temperature=TEMP)
    scheduler   = torch.optim.lr_scheduler.LinearLR(
                      optimizer, start_factor=1.0, end_factor=0.1,
                      total_iters=EPOCHS)

    best_val_f1 = 0.0
    history     = []

    print(f"\n{'Epoch':>5} {'TrainLoss':>10} {'TrainF1':>9} {'ValLoss':>9} {'ValF1':>8}  Best?")
    print("-" * 55)

    for epoch in range(1, EPOCHS + 1):
        tr = run_epoch(model, train_loader, optimizer, supcon_loss, train=True)
        va = run_epoch(model, val_loader,   optimizer, supcon_loss, train=False)
        scheduler.step()
        is_best = va['binary_f1'] > best_val_f1
        if is_best:
            best_val_f1 = va['binary_f1']
            torch.save({'model_state_dict': model.state_dict()}, CKPT_PATH)
        print(f"{epoch:>5} {tr['loss']:>10.4f} {tr['binary_f1']:>9.4f} "
              f"{va['loss']:>9.4f} {va['binary_f1']:>8.4f}  {'*' if is_best else ''}")
        history.append({'epoch': epoch, 'train': tr, 'val': va})

    print(f"\nLoading best checkpoint (val F1={best_val_f1:.4f})...")
    ckpt  = torch.load(CKPT_PATH, map_location=DEVICE)
    state = ckpt.get('model_state_dict', ckpt)
    if all(k.startswith('module.') for k in state.keys()):
        state = {k[7:]: v for k, v in state.items()}
    model.module.load_state_dict(state) if hasattr(model, 'module') else model.load_state_dict(state)

    te = run_epoch(model, test_loader, optimizer, supcon_loss, train=False)

    label = 'SupCon only (no aux heads)' if AUX_WEIGHT == 0.0 else f'Full model (aux_wt={AUX_WEIGHT})'
    print("\n" + "=" * 60)
    print(f"TEST RESULTS  (λ={LAM}, aux_weight={AUX_WEIGHT} — {label})")
    print("=" * 60)
    print(f"Binary macro-F1:  {te['binary_f1']:.4f}")
    print(f"Binary pos-F1:    {te['binary_f1_pos']:.4f}")
    print("\nPer domain (macro-F1):")
    for prefix in ['cs', 'mono']:
        k = f'f1_{prefix}'
        if k in te:
            print(f"  {prefix:<20} {te[k]:.4f}")
    print("\nPer category (accuracy):")
    for cat in CATEGORIES:
        k = f'acc_{cat}'
        if k in te:
            print(f"  {cat:<20} {te[k]:.4f}")
    print("\nPer language pair (macro-F1):")
    for lp in LANG_PAIRS:
        k = f'f1_{lp}'
        if k in te:
            print(f"  {lp:<10} {te[k]:.4f}")

    with open(RESULTS_PATH, 'w') as f:
        json.dump({
            'lam': LAM, 'aux_weight': AUX_WEIGHT,
            'note': label,
            'test': te, 'history': history
        }, f, indent=2)
    print(f"\nResults saved → {RESULTS_PATH}")

if __name__ == '__main__':
    main()
