"""
CogSwitch Adversarial Robustness Evaluation — λ=0.7, data/final_v2
Table XII equivalent

Run:
    cd /home/nmunir/cogswitch
    source venv/bin/activate
    python3 adversarial_robustness.py 2>&1 | tee adversarial_log.txt
"""

import json, random
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import AutoTokenizer, AutoModel

CHECKPOINT  = "/home/nmunir/cogswitch/checkpoints/best_lam0.7.pt"
TEST_CSV    = "/home/nmunir/cogswitch/data/final_v2/test.csv"
OUT_JSON    = "/home/nmunir/cogswitch/adversarial_results.json"
OUT_TXT     = "/home/nmunir/cogswitch/adversarial_results.txt"
MODEL_BASE  = "xlm-roberta-base"
CS_PAIRS    = ["en_hi", "en_ar", "en_es", "en_zh"]
MAX_SAMPLES = 300
SEED        = 42
DEVICE      = "cuda" if torch.cuda.is_available() else "cpu"

# ── Model ─────────────────────────────────────────────────────────────────────
class CogSwitchModel(nn.Module):
    def __init__(self, model_name=MODEL_BASE, n_persuasion=5):
        super().__init__()
        self.encoder    = AutoModel.from_pretrained(model_name)
        hidden          = self.encoder.config.hidden_size   # 768
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

# ── Load model ────────────────────────────────────────────────────────────────
print(f"Device: {DEVICE}")
print("Loading model...")
tokenizer = AutoTokenizer.from_pretrained(MODEL_BASE)
model     = CogSwitchModel().to(DEVICE)

ckpt  = torch.load(CHECKPOINT, map_location=DEVICE)
state = ckpt.get("model_state_dict", ckpt)
if all(k.startswith("module.") for k in state.keys()):
    state = {k[7:]: v for k, v in state.items()}
missing, unexpected = model.load_state_dict(state, strict=True)
assert len(missing) == 0,    f"Missing keys: {missing}"
assert len(unexpected) == 0, f"Unexpected keys: {unexpected}"
model.eval()
print("Model loaded OK")

# ── Predict helper ────────────────────────────────────────────────────────────
def predict(texts, batch_size=64):
    preds, probs = [], []
    for i in range(0, len(texts), batch_size):
        batch = texts[i:i+batch_size]
        enc   = tokenizer(batch, return_tensors="pt", truncation=True,
                          padding=True, max_length=128).to(DEVICE)
        with torch.no_grad():
            logits, _, _ = model(**enc)
        p = torch.softmax(logits, dim=-1)
        preds.extend(torch.argmax(logits, dim=-1).cpu().tolist())
        probs.extend(p[:, 1].cpu().tolist())
    return preds, probs

# ── Load & sample ─────────────────────────────────────────────────────────────
print("\nLoading test data...")
df = pd.read_csv(TEST_CSV)
print(f"Total rows: {len(df)}")
print("Columns:", df.columns.tolist())

# Filter: CS phishing only
cs_phish = df[(df["lang_pair"].isin(CS_PAIRS)) & (df["label"] == 1)].copy()
print(f"\nCS phishing rows available: {len(cs_phish)}")
print(cs_phish["lang_pair"].value_counts().to_string())

# Proportional stratified sample of 300
cs_phish = cs_phish.sample(min(MAX_SAMPLES, len(cs_phish)),
                            random_state=SEED).reset_index(drop=True)
print(f"\nSampled: {len(cs_phish)}")
print(cs_phish["lang_pair"].value_counts().to_string())

# ── Baseline: verify all 300 are correctly classified ─────────────────────────
orig_preds, orig_probs = predict(cs_phish["text"].tolist())
n_correct = sum(p == 1 for p in orig_preds)
print(f"\nBaseline: {n_correct}/{len(cs_phish)} correctly classified as phishing")
print(f"Avg confidence (P(phishing)): {sum(orig_probs)/len(orig_probs):.4f}")

if n_correct < len(cs_phish):
    wrong_idx = [i for i, p in enumerate(orig_preds) if p != 1]
    print(f"WARNING: {len(wrong_idx)} samples NOT correctly classified before attack.")
    print("These will be excluded from ASR calculation (only count flips from correct).")

# ── Attack 1: Homoglyph substitution (15 % of eligible Latin chars) ───────────
# Cyrillic visually identical lookalikes
HOMO = {
    'a': 'а', 'e': 'е', 'o': 'о', 'p': 'р', 'c': 'с',
    'x': 'х', 'i': 'і', 'A': 'А', 'E': 'Е', 'O': 'О',
    'B': 'В', 'H': 'Н', 'K': 'К', 'M': 'М', 'T': 'Т',
}

def homoglyph_attack(text, rate=0.15, seed=0):
    rng     = random.Random(seed)
    chars   = list(text)
    targets = [i for i, c in enumerate(chars) if c in HOMO]
    if not targets:
        return text
    n_to_replace = min(max(1, int(len(targets) * rate)), len(targets))
    chosen       = rng.sample(targets, n_to_replace)
    for i in chosen:
        chars[i] = HOMO[chars[i]]
    return "".join(chars)

hg_texts = [homoglyph_attack(t, rate=0.15, seed=i)
            for i, t in enumerate(cs_phish["text"])]

chars_changed = [sum(a != b for a, b in zip(o, h))
                 for o, h in zip(cs_phish["text"], hg_texts)]
print(f"\nHomoglyph: avg chars substituted per sample: "
      f"{sum(chars_changed)/len(chars_changed):.1f}")

hg_preds, hg_probs = predict(hg_texts)

# ASR = samples that were correct before AND flipped after
hg_flipped = sum(1 for o, h in zip(orig_preds, hg_preds) if o == 1 and h == 0)
n_eligible  = n_correct                    # denominator: originally-correct
hg_asr      = hg_flipped / n_eligible if n_eligible else 0
print(f"Homoglyph ASR: {hg_asr:.4f}  ({hg_flipped}/{n_eligible} flipped)")

hg_by_pair = {}
for pair in CS_PAIRS:
    mask  = cs_phish["lang_pair"] == pair
    iloc  = [cs_phish.index.get_loc(i) for i in cs_phish[mask].index]
    succ  = sum(1 for i in iloc if orig_preds[i] == 1 and hg_preds[i] == 0)
    elig  = sum(1 for i in iloc if orig_preds[i] == 1)
    n     = len(iloc)
    cc    = (sum(chars_changed[i] for i in iloc) / n) if n else 0
    hg_by_pair[pair] = {
        "flipped": succ, "eligible": elig, "total": n,
        "asr": round(succ / elig, 4) if elig else 0.0,
        "avg_chars_changed": round(cc, 1),
    }
    print(f"  {pair}: ASR={hg_by_pair[pair]['asr']:.4f}  "
          f"({succ}/{elig})  avg_changed={cc:.1f}")

# ── Attack 2: Character perturbation (10 % rate: insert / delete / swap) ──────
def char_perturb_attack(text, rate=0.10, seed=0):
    rng   = random.Random(seed)
    chars = list(text)
    n_ops = max(1, int(len(chars) * rate))
    for _ in range(n_ops):
        if not chars:
            break
        op = rng.choice(["insert", "delete", "swap"])
        i  = rng.randint(0, len(chars) - 1)
        if op == "insert":
            chars.insert(i, rng.choice("abcdefghijklmnopqrstuvwxyz"))
        elif op == "delete" and len(chars) > 1:
            chars.pop(i)
        elif op == "swap" and i < len(chars) - 1:
            chars[i], chars[i + 1] = chars[i + 1], chars[i]
    return "".join(chars)

cp_texts = [char_perturb_attack(t, rate=0.10, seed=i)
            for i, t in enumerate(cs_phish["text"])]

cp_preds, cp_probs = predict(cp_texts)

cp_flipped = sum(1 for o, h in zip(orig_preds, cp_preds) if o == 1 and h == 0)
cp_asr     = cp_flipped / n_eligible if n_eligible else 0
print(f"\nChar-Perturb ASR: {cp_asr:.4f}  ({cp_flipped}/{n_eligible} flipped)")

cp_by_pair = {}
for pair in CS_PAIRS:
    mask  = cs_phish["lang_pair"] == pair
    iloc  = [cs_phish.index.get_loc(i) for i in cs_phish[mask].index]
    succ  = sum(1 for i in iloc if orig_preds[i] == 1 and cp_preds[i] == 0)
    elig  = sum(1 for i in iloc if orig_preds[i] == 1)
    n     = len(iloc)
    cp_by_pair[pair] = {
        "flipped": succ, "eligible": elig, "total": n,
        "asr": round(succ / elig, 4) if elig else 0.0,
    }
    print(f"  {pair}: ASR={cp_by_pair[pair]['asr']:.4f}  ({succ}/{elig})")

# ── Print Table XII equivalent ────────────────────────────────────────────────
lines = []
lines.append("")
lines.append("=" * 72)
lines.append("TABLE XII — Adversarial Robustness  |  CogSwitch λ=0.7")
lines.append(f"  {len(cs_phish)} CS phishing messages sampled from data/final_v2 (seed={SEED})")
lines.append(f"  Baseline: {n_correct}/{len(cs_phish)} correctly classified (avg conf "
             f"{sum(orig_probs)/len(orig_probs):.4f})")
lines.append("=" * 72)

header = f"{'Language Pair':<16}{'N':>5}{'  Homoglyph 15%':>17}{'  Char-Perturb 10%':>19}"
lines.append(header)
lines.append("-" * 72)

for pair in CS_PAIRS:
    hg = hg_by_pair[pair]
    cp = cp_by_pair[pair]
    hg_str = f"{hg['flipped']}/{hg['eligible']} ({hg['asr']*100:.2f}%)"
    cp_str = f"{cp['flipped']}/{cp['eligible']} ({cp['asr']*100:.2f}%)"
    lines.append(f"{pair:<16}{hg['total']:>5}  {hg_str:>15}  {cp_str:>17}")

lines.append("-" * 72)
# Overall row
hg_tot_str = f"{hg_flipped}/{n_eligible} ({hg_asr*100:.2f}%)"
cp_tot_str = f"{cp_flipped}/{n_eligible} ({cp_asr*100:.2f}%)"
lines.append(f"{'Overall':<16}{len(cs_phish):>5}  {hg_tot_str:>15}  {cp_tot_str:>17}")
lines.append("=" * 72)
lines.append("ASR = Attack Success Rate (flipped / originally-correct)")
lines.append("")

table_str = "\n".join(lines)
print(table_str)

# ── Save JSON + TXT ───────────────────────────────────────────────────────────
results = {
    "checkpoint": CHECKPOINT,
    "test_csv":   TEST_CSV,
    "lambda":     0.7,
    "n_samples":  len(cs_phish),
    "seed":       SEED,
    "lang_pair_counts": cs_phish["lang_pair"].value_counts().to_dict(),
    "baseline": {
        "n_correct":    n_correct,
        "avg_conf":     round(sum(orig_probs)/len(orig_probs), 4),
    },
    "Homoglyph_15pct": {
        "rate":       0.15,
        "flipped":    hg_flipped,
        "eligible":   n_eligible,
        "asr":        round(hg_asr, 4),
        "asr_pct":    round(hg_asr * 100, 2),
        "by_pair":    hg_by_pair,
    },
    "CharPerturb_10pct": {
        "rate":       0.10,
        "flipped":    cp_flipped,
        "eligible":   n_eligible,
        "asr":        round(cp_asr, 4),
        "asr_pct":    round(cp_asr * 100, 2),
        "by_pair":    cp_by_pair,
    },
}

with open(OUT_JSON, "w") as f:
    json.dump(results, f, indent=2)
print(f"Saved JSON -> {OUT_JSON}")

with open(OUT_TXT, "w", encoding="utf-8") as f:
    f.write(table_str)
    f.write("\n\nFull JSON:\n")
    f.write(json.dumps(results, indent=2))
print(f"Saved TXT  -> {OUT_TXT}")
