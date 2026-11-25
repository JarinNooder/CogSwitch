"""
CogSwitch Training Script
XLM-RoBERTa + Supervised Contrastive Loss + Multi-task Persuasion Heads
Loss: L = (1-lambda)*CE + lambda*SupCon
Run: python3 -u train_cogswitch.py --lam 0.7 2>&1 | tee logs/train_lam07.log
"""

import os, argparse, json, random
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from transformers import AutoTokenizer, AutoModel
from sklearn.metrics import f1_score, classification_report, accuracy_score
from datetime import datetime

# ── Args ──────────────────────────────────────────────────────────────────────
parser = argparse.ArgumentParser()
parser.add_argument('--lam',        type=float, default=0.7,   help='SupCon weight')
parser.add_argument('--lr',         type=float, default=2e-5)
parser.add_argument('--epochs',     type=int,   default=10)
parser.add_argument('--batch_size', type=int,   default=32)
parser.add_argument('--max_len',    type=int,   default=128)
parser.add_argument('--temp',       type=float, default=0.1,  help='SupCon temperature')
parser.add_argument('--seed',       type=int,   default=42)
parser.add_argument('--model_name', type=str,   default='xlm-roberta-base')
parser.add_argument('--data_dir',   type=str,   default='data/final_v2')
parser.add_argument('--out_dir',    type=str,   default='checkpoints')
parser.add_argument('--aux_weight', type=float, default=0.1, help='Persuasion aux loss weight')
args = parser.parse_args()

# ── Reproducibility ───────────────────────────────────────────────────────────
random.seed(args.seed)
np.random.seed(args.seed)
torch.manual_seed(args.seed)
torch.cuda.manual_seed_all(args.seed)

os.makedirs(args.out_dir, exist_ok=True)
os.makedirs('logs', exist_ok=True)

DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
N_GPU  = torch.cuda.device_count()
print(f"Using {N_GPU} GPU(s): {[torch.cuda.get_device_name(i) for i in range(N_GPU)]}")
print(f"λ={args.lam}  lr={args.lr}  epochs={args.epochs}  batch={args.batch_size}")

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
        # Persuasion: NaN -> -1 (masked)
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

# ── Supervised Contrastive Loss ───────────────────────────────────────────────
class SupConLoss(nn.Module):
    def __init__(self, temperature=0.1):
        super().__init__()
        self.temperature = temperature

    def forward(self, features, labels):
        """
        features: (B, D) L2-normalized
        labels:   (B,)
        """
        device = features.device
        B = features.shape[0]
        if B < 2:
            return torch.tensor(0.0, device=device)

        # Cosine similarity matrix
        sim = torch.matmul(features, features.T) / self.temperature  # (B, B)

        # Mask out self
        mask_self = torch.eye(B, dtype=torch.bool, device=device)
        sim = sim.masked_fill(mask_self, float('-inf'))

        # Positive mask: same label, not self
        labels = labels.view(-1, 1)
        mask_pos = (labels == labels.T) & ~mask_self  # (B, B)

        # Skip if no positives in batch
        if mask_pos.sum() == 0:
            return torch.tensor(0.0, device=device)

        log_prob = F.log_softmax(sim, dim=1)
        # Mean over positives per anchor, then mean over anchors with >=1 positive
        n_pos = mask_pos.sum(dim=1).float()
        has_pos = n_pos > 0
        positive_log_prob = log_prob.masked_fill(~mask_pos, 0.0)
        loss = -(positive_log_prob.sum(dim=1)[has_pos] / n_pos[has_pos]).mean()
        return loss

# ── Model ─────────────────────────────────────────────────────────────────────
class CogSwitchModel(nn.Module):
    def __init__(self, model_name, n_persuasion=5):
        super().__init__()
        self.encoder    = AutoModel.from_pretrained(model_name)
        hidden          = self.encoder.config.hidden_size  # 768

        # Projection head for SupCon (128-dim L2-normalized)
        self.projector  = nn.Sequential(
            nn.Linear(hidden, hidden),
            nn.ReLU(),
            nn.Linear(hidden, 128)
        )

        # Primary classification head (binary: phishing vs ham)
        self.classifier = nn.Linear(hidden, 2)

        # Auxiliary persuasion heads (one per label)
        self.aux_heads  = nn.ModuleList([nn.Linear(hidden, 2) for _ in range(n_persuasion)])

    def forward(self, input_ids, attention_mask):
        out     = self.encoder(input_ids=input_ids, attention_mask=attention_mask)
        cls     = out.last_hidden_state[:, 0, :]          # (B, 768)
        logits  = self.classifier(cls)                     # (B, 2)
        proj    = F.normalize(self.projector(cls), dim=1)  # (B, 128)
        aux     = [head(cls) for head in self.aux_heads]   # list of (B, 2)
        return logits, proj, aux, cls

# ── Training utilities ────────────────────────────────────────────────────────
def masked_bce_loss(aux_logits, persuasion_labels):
    """Binary CE loss on persuasion heads, masked where label == -1."""
    total_loss = 0.0
    n_valid    = 0
    for i, logits in enumerate(aux_logits):
        labels = persuasion_labels[:, i]       # (B,)
        mask   = labels != -1
        if mask.sum() == 0:
            continue
        loss = F.cross_entropy(logits[mask], labels[mask])
        total_loss += loss
        n_valid    += 1
    return total_loss / n_valid if n_valid > 0 else torch.tensor(0.0, device=aux_logits[0].device)

def compute_metrics(all_labels, all_preds, all_cats, all_langs):
    results = {}

    # 1. Binary F1
    results['binary_f1']       = f1_score(all_labels, all_preds, average='macro')
    results['binary_f1_pos']   = f1_score(all_labels, all_preds, pos_label=1, average='binary')
    results['accuracy']        = accuracy_score(all_labels, all_preds)

    # 2. Per-category F1: cs_phishing+cs_ham and mono_phishing+mono_ham
    for prefix in ['cs', 'mono']:
        idx = [i for i, c in enumerate(all_cats) if c.startswith(prefix)]
        if not idx:
            continue
        y_t = [all_labels[i] for i in idx]
        y_p = [all_preds[i]  for i in idx]
        results[f'f1_{prefix}'] = f1_score(y_t, y_p, average='macro', zero_division=0)
    # Also keep individual category accuracy
    for cat in CATEGORIES:
        idx = [i for i, c in enumerate(all_cats) if c == cat]
        if not idx:
            continue
        correct = sum(all_labels[i] == all_preds[i] for i in idx)
        results[f'acc_{cat}'] = correct / len(idx)

    # 3. Per-language-pair F1
    for lp in LANG_PAIRS:
        idx = [i for i, l in enumerate(all_langs) if l == lp]
        if not idx:
            continue
        y_t = [all_labels[i] for i in idx]
        y_p = [all_preds[i]  for i in idx]
        results[f'f1_{lp}'] = f1_score(y_t, y_p, average='macro', zero_division=0)

    return results

def run_epoch(model, loader, optimizer, supcon_loss_fn, lam, aux_weight, class_weights, train=True):
    model.train() if train else model.eval()
    total_loss = 0.0
    all_labels, all_preds, all_cats, all_langs = [], [], [], []

    ctx = torch.enable_grad() if train else torch.no_grad()
    with ctx:
        for batch in loader:
            input_ids  = batch['input_ids'].to(DEVICE)
            attn_mask  = batch['attention_mask'].to(DEVICE)
            labels     = batch['label'].to(DEVICE)
            persuasion = batch['persuasion'].to(DEVICE)

            logits, proj, aux, _ = model(input_ids, attn_mask)

            # Primary CE loss (weighted for class imbalance)
            ce_loss  = F.cross_entropy(logits, labels, weight=class_weights)

            # SupCon loss
            sc_loss  = supcon_loss_fn(proj, labels)

            # Auxiliary persuasion loss
            ax_loss  = masked_bce_loss(aux, persuasion)

            if not torch.isfinite(sc_loss):
                raise RuntimeError(f"SupCon loss is not finite: {sc_loss.item()}")
            ax_loss  = ax_loss if not torch.isnan(ax_loss) else torch.tensor(0.0, device=labels.device)
            loss = (1 - lam) * ce_loss + lam * sc_loss + aux_weight * ax_loss
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
    tokenizer = AutoTokenizer.from_pretrained(args.model_name)

    print("Loading datasets...")
    train_ds = CogSwitchDataset(f'{args.data_dir}/train.csv', tokenizer, args.max_len)
    val_ds   = CogSwitchDataset(f'{args.data_dir}/val.csv',   tokenizer, args.max_len)
    test_ds  = CogSwitchDataset(f'{args.data_dir}/test.csv',  tokenizer, args.max_len)

    train_loader = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True,  num_workers=4, pin_memory=True)
    val_loader   = DataLoader(val_ds,   batch_size=args.batch_size, shuffle=False, num_workers=4, pin_memory=True)
    test_loader  = DataLoader(test_ds,  batch_size=args.batch_size, shuffle=False, num_workers=4, pin_memory=True)

    print("Loading model...")
    model = CogSwitchModel(args.model_name)
    if N_GPU > 1:
        model = nn.DataParallel(model)
    model = model.to(DEVICE)

    optimizer     = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=0.01)
    supcon_loss   = SupConLoss(temperature=args.temp)
    scheduler     = torch.optim.lr_scheduler.LinearLR(
                        optimizer, start_factor=1.0, end_factor=0.1,
                        total_iters=args.epochs)

    # Class weights to handle ham/phishing imbalance (computed from train set)
    class_weights = torch.tensor([0.77748957, 1.40093477], dtype=torch.float).to(DEVICE)

    best_val_f1   = 0.0
    best_ckpt     = os.path.join(args.out_dir, f'best_lam{args.lam}.pt')
    history       = []

    print(f"\n{'Epoch':>5} {'TrainLoss':>10} {'TrainF1':>9} {'ValLoss':>9} {'ValF1':>8}  Best?")
    print("-" * 55)

    for epoch in range(1, args.epochs + 1):
        tr = run_epoch(model, train_loader, optimizer, supcon_loss, args.lam, args.aux_weight, class_weights, train=True)
        va = run_epoch(model, val_loader,   optimizer, supcon_loss, args.lam, args.aux_weight, class_weights, train=False)
        scheduler.step()

        is_best = va['binary_f1'] > best_val_f1
        if is_best:
            best_val_f1 = va['binary_f1']
            torch.save(model.state_dict(), best_ckpt)

        print(f"{epoch:>5} {tr['loss']:>10.4f} {tr['binary_f1']:>9.4f} "
              f"{va['loss']:>9.4f} {va['binary_f1']:>8.4f}  {'*' if is_best else ''}")

        history.append({'epoch': epoch, 'train': tr, 'val': va})

    # ── Test evaluation ───────────────────────────────────────────────────────
    print(f"\nLoading best checkpoint (val F1={best_val_f1:.4f})...")
    model.load_state_dict(torch.load(best_ckpt))
    te = run_epoch(model, test_loader, optimizer, supcon_loss, args.lam, args.aux_weight, class_weights, train=False)

    print("\n" + "="*60)
    print(f"TEST RESULTS  (λ={args.lam})")
    print("="*60)
    print(f"Binary macro-F1:  {te['binary_f1']:.4f}")
    print(f"Accuracy:         {te['accuracy']:.4f}")
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
    print("\nPer language pair:")
    for lp in LANG_PAIRS:
        k = f'f1_{lp}'
        if k in te:
            print(f"  {lp:<10} {te[k]:.4f}")

    # Save results
    results_path = os.path.join(args.out_dir, f'results_lam{args.lam}.json')
    with open(results_path, 'w') as f:
        json.dump({'args': vars(args), 'test': te, 'history': history}, f, indent=2)
    print(f"\nResults saved → {results_path}")

if __name__ == '__main__':
    main()
