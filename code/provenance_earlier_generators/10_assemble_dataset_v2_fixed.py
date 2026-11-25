"""
Script 10: Dataset Assembly v2 (CogSwitch v6)
Builds the final CogSwitch dataset from:
  - Synthetic CS phishing (v6 validated, all 4 pairs) — source-level split preserved
  - Real CS ham (data/cs_ham/)
  - Monolingual phishing (English, Arabic, Spanish, Chinese, Hinglish)
  - Monolingual ham (English + multilingual)

Key difference from v1: CS phishing uses pre-split v6 files (train/val/test)
to preserve source-level splitting and prevent data leakage. Non-CS data
is split 70/15/15 afterwards and merged with the pre-split CS rows.

Output schema:
  text, label, lang_pair, category, source,
  persuasion_authority, persuasion_social_proof,
  persuasion_liking, persuasion_commitment, persuasion_distraction
  (persuasion cols are NaN for non-SpAPhish rows)

label: 1=phishing, 0=ham
category: cs_phishing | cs_ham | mono_phishing | mono_ham

Saves:
  data/final_v2/full_dataset.csv
  data/final_v2/train.csv  (CS pre-split + 70% of rest)
  data/final_v2/val.csv    (CS pre-split + 15% of rest)
  data/final_v2/test.csv   (CS pre-split + 15% of rest)

Run from /home/nmunir/cogswitch with venv active.
"""

import os, re, random
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from collections import Counter

random.seed(42)
np.random.seed(42)

# ─── Mixing metric helpers (same logic as 04c_validate_v2.py) ─────────────────
def tokenize(text):
    return re.findall(r"[\w']+", str(text).lower())

def detect_lang_simple(token, pair):
    """Very simple heuristic: non-ASCII = L2, ASCII = English."""
    return 'en' if token.isascii() else 'l2'

def compute_cmi(text):
    tokens = tokenize(text)
    if len(tokens) < 2:
        return 0.0
    langs = ['en' if t.isascii() else 'l2' for t in tokens]
    counts = Counter(langs)
    n = len(langs)
    max_lang = max(counts.values())
    switches = sum(1 for i in range(1, len(langs)) if langs[i] != langs[i-1])
    cmi = 1.0 - (max_lang / n) + (switches / (2 * (n - 1))) if n > 1 else 0.0
    return float(np.clip(cmi, 0, 1))

def compute_i_index(text):
    tokens = tokenize(text)
    if len(tokens) < 2:
        return 0.0
    langs = ['en' if t.isascii() else 'l2' for t in tokens]
    switches = sum(1 for i in range(1, len(langs)) if langs[i] != langs[i-1])
    return switches / (len(langs) - 1)

def compute_m_index(text):
    tokens = tokenize(text)
    if not tokens:
        return 0.0
    langs = ['en' if t.isascii() else 'l2' for t in tokens]
    counts = Counter(langs)
    n = len(langs)
    k = len(counts)
    if k < 2:
        return 0.0
    return (1 - sum((v/n)**2 for v in counts.values())) / (1 - 1/k)

def is_valid_cs(text, cmi_thresh=0.1, i_thresh=0.1, m_thresh=0.2):
    cmi = compute_cmi(text)
    i   = compute_i_index(text)
    m   = compute_m_index(text)
    return cmi > cmi_thresh and i > i_thresh and m > m_thresh

def filter_cs(df, label=""):
    before = len(df)
    mask = df['text'].apply(is_valid_cs)
    df = df[mask].reset_index(drop=True)
    print(f"  {label}: {before} → {len(df)} after CMI filter ({before-len(df)} removed)")
    return df

OUT_DIR = "data/final_v2"
os.makedirs(OUT_DIR, exist_ok=True)

PERSUASION_COLS = [
    'persuasion_authority', 'persuasion_social_proof',
    'persuasion_liking', 'persuasion_commitment', 'persuasion_distraction'
]

# Rows that have a designated split (CS phishing v6)
cs_train_parts = []
cs_val_parts   = []
cs_test_parts  = []

# Rows to be split randomly (everything else)
other_parts = []

# ─── 1. Synthetic CS phishing (v6 — source-level pre-split) ───────────────────
print("=== Loading synthetic CS phishing (v6) ===")
for pair in ["en_hi", "en_ar", "en_es", "en_zh"]:
    for split, bucket in [("train", cs_train_parts),
                           ("val",   cs_val_parts),
                           ("test",  cs_test_parts)]:
        path = f"data/validated_v6/{pair}_cs_v6_{split}_validated.csv"
        if not os.path.exists(path):
            print(f"  MISSING: {path}")
            continue
        df = pd.read_csv(path)
        # keep text column; lang_pair may already be present
        if 'text' not in df.columns:
            # try common alternatives
            tcol = next((c for c in df.columns
                         if 'text' in c.lower() or 'message' in c.lower()), df.columns[0])
            df = df.rename(columns={tcol: 'text'})
        df = df[['text']].copy()
        df['lang_pair'] = pair
        df['label']    = 1
        df['category'] = 'cs_phishing'
        df['source']   = 'llama3_cs_v6'
        for c in PERSUASION_COLS:
            df[c] = np.nan
        bucket.append(df)
    # report total per pair
    n = sum(len(b[-1]) for b in [cs_train_parts, cs_val_parts, cs_test_parts]
            if b and b[-1]['lang_pair'].iloc[0] == pair)

cs_train = pd.concat(cs_train_parts, ignore_index=True) if cs_train_parts else pd.DataFrame()
cs_val   = pd.concat(cs_val_parts,   ignore_index=True) if cs_val_parts   else pd.DataFrame()
cs_test  = pd.concat(cs_test_parts,  ignore_index=True) if cs_test_parts  else pd.DataFrame()

print(f"  CS phishing — train: {len(cs_train)}  val: {len(cs_val)}  test: {len(cs_test)}")
for pair in ["en_hi", "en_ar", "en_es", "en_zh"]:
    n = sum(len(df[df['lang_pair']==pair]) for df in [cs_train, cs_val, cs_test])
    print(f"    {pair}: {n} total")

# ─── 2. Real CS ham ───────────────────────────────────────────────────────────
print("\n=== Loading real CS ham ===")
CS_HAM = {
    "en_hi": "data/cs_ham/en_hi_ham.csv",
    "en_ar": "data/cs_ham/en_ar_ham.csv",
    "en_es": "data/cs_ham/en_es_ham.csv",
    "en_zh": "data/cs_ham/en_zh_ham.csv",
}
for pair, path in CS_HAM.items():
    if not os.path.exists(path):
        print(f"  MISSING: {path}")
        continue
    df = pd.read_csv(path)
    tcol = next((c for c in df.columns
                 if 'text' in c.lower() or 'transcript' in c.lower()), df.columns[0])
    df = df[[tcol]].rename(columns={tcol: 'text'}).dropna()
    df['text'] = df['text'].astype(str).str.strip()
    df = df[df['text'].str.len() > 10]
    if len(df) > 2000:
        df = df.sample(2000, random_state=42)
    df['label']     = 0
    df['lang_pair'] = pair
    df['category']  = 'cs_ham'
    df['source']    = f'real_cs_ham_{pair}'
    for c in PERSUASION_COLS:
        df[c] = np.nan
    other_parts.append(df)
    print(f"  {pair}: {len(df)} rows")

# ─── 3. Monolingual phishing ──────────────────────────────────────────────────
print("\n=== Loading monolingual phishing ===")
mono_phish_parts = []

# English phishing pool
if os.path.exists("data/raw/phishing.csv"):
    df = pd.read_csv("data/raw/phishing.csv")
    tcol = next((c for c in df.columns
                 if 'text' in c.lower() or 'message' in c.lower() or 'sms' in c.lower()),
                df.columns[0])
    lcol = next((c for c in df.columns
                 if 'label' in c.lower() or 'spam' in c.lower() or 'class' in c.lower()), None)
    if lcol:
        sub = df[df[lcol].isin([1, 'spam', 'phishing', 'Phishing', 'smishing'])]
        if len(sub) == 0: sub = df
    else:
        sub = df
    texts = sub[tcol].dropna().astype(str).tolist()
    mono_phish_parts.append(pd.DataFrame({'text': texts, 'source': 'phishing_csv'}))
    print(f"  phishing.csv: {len(texts)} rows")

# SpAPhish Spanish — WITH persuasion labels
if os.path.exists("data/raw/spaphish_spanish.csv"):
    df = pd.read_csv("data/raw/spaphish_spanish.csv")
    sub = df[df['label'] == 1].copy() if 'label' in df.columns else df.copy()
    sub = sub.rename(columns={
        'authority':                          'persuasion_authority',
        'social_proof':                       'persuasion_social_proof',
        'liking_similarity_deception':        'persuasion_liking',
        'commitment_integrity_reciprocation': 'persuasion_commitment',
        'distraction':                        'persuasion_distraction',
    })
    sub['source'] = 'spaphish'
    keep_cols = ['text', 'source'] + PERSUASION_COLS
    sub = sub[[c for c in keep_cols if c in sub.columns]]
    for c in PERSUASION_COLS:
        if c not in sub.columns:
            sub[c] = np.nan
    mono_phish_parts.append(sub)
    print(f"  spaphish_spanish.csv: {len(sub)} phishing rows (with persuasion labels)")

# Arabic spam
for fname, label_vals in [
    ("data/raw/arabic_sms_spam.csv",   [1, 'spam', 'Spam']),
    ("data/raw/arabic_spam_tweets.csv", [1, 'spam', 'Spam']),
]:
    if not os.path.exists(fname): continue
    df = pd.read_csv(fname)
    tcol = next((c for c in df.columns
                 if 'text' in c.lower() or 'tweet' in c.lower() or 'message' in c.lower()),
                df.columns[0])
    lcol = next((c for c in df.columns
                 if 'label' in c.lower() or 'spam' in c.lower()), None)
    sub = df[df[lcol].isin(label_vals)] if lcol else df
    texts = sub[tcol].dropna().astype(str).tolist()
    mono_phish_parts.append(pd.DataFrame({'text': texts, 'source': os.path.basename(fname)}))
    print(f"  {os.path.basename(fname)}: {len(texts)} rows")

# Chinese fraud
if os.path.exists("data/raw/chifraud_chinese.csv"):
    df = pd.read_csv("data/raw/chifraud_chinese.csv")
    print(f"  chifraud columns: {list(df.columns)[:6]}")
    tcol = next((c for c in df.columns
                 if 'text' in c.lower() or 'content' in c.lower() or 'message' in c.lower()),
                df.columns[0])
    lcol = next((c for c in df.columns
                 if 'label' in c.lower() or 'fraud' in c.lower() or 'class' in c.lower()), None)
    if lcol:
        print(f"    label dist: {df[lcol].value_counts().head().to_dict()}")
        sub = df[df[lcol].isin([1, 'fraud', 'Fraud', 'spam', 'phishing', '1'])]
        if len(sub) == 0: sub = df
    else:
        sub = df
    texts = sub[tcol].dropna().astype(str).tolist()
    mono_phish_parts.append(pd.DataFrame({'text': texts, 'source': 'chifraud'}))
    print(f"  chifraud_chinese.csv: {len(texts)} rows")

# Hinglish phishing (real)
for fname in ["data/raw/indian_cyber_scam_hinglish.csv", "data/raw/scamshield_hinglish.csv"]:
    if not os.path.exists(fname): continue
    df = pd.read_csv(fname)
    print(f"  {os.path.basename(fname)} columns: {list(df.columns)[:6]}")
    tcol = next((c for c in df.columns
                 if 'text' in c.lower() or 'message' in c.lower() or 'sms' in c.lower()),
                df.columns[0])
    lcol = next((c for c in df.columns
                 if 'label' in c.lower() or 'spam' in c.lower() or 'class' in c.lower()), None)
    if lcol:
        print(f"    label dist: {df[lcol].value_counts().to_dict()}")
        sub = df[df[lcol].isin([1, 'spam', 'phishing', 'scam', 'Scam'])]
        if len(sub) == 0: sub = df
    else:
        sub = df
    texts = sub[tcol].dropna().astype(str).tolist()
    mono_phish_parts.append(pd.DataFrame({'text': texts, 'source': os.path.basename(fname)}))
    print(f"  {os.path.basename(fname)}: {len(texts)} rows")

# Combine mono phishing
mono_phish_df = pd.concat(mono_phish_parts, ignore_index=True)
mono_phish_df['text'] = mono_phish_df['text'].astype(str).str.strip()
mono_phish_df = mono_phish_df[mono_phish_df['text'].str.len() > 10]
for c in PERSUASION_COLS:
    if c not in mono_phish_df.columns:
        mono_phish_df[c] = np.nan
mono_phish_df['label']     = 1
mono_phish_df['lang_pair'] = 'en'
mono_phish_df['category']  = 'mono_phishing'
# Sample to cap at 5000 (preserve all spaphish rows)
if len(mono_phish_df) > 5000:
    spaphish = mono_phish_df[mono_phish_df['source'] == 'spaphish']
    rest     = mono_phish_df[mono_phish_df['source'] != 'spaphish']
    n_rest   = max(0, 5000 - len(spaphish))
    rest     = rest.sample(min(n_rest, len(rest)), random_state=42)
    mono_phish_df = pd.concat([spaphish, rest], ignore_index=True)
print(f"\n  Total mono phishing: {len(mono_phish_df)} rows")
print(f"  SpAPhish rows (with persuasion): {(mono_phish_df['source']=='spaphish').sum()}")
other_parts.append(mono_phish_df)

# ─── 4. Monolingual ham ───────────────────────────────────────────────────────
print("\n=== Loading monolingual ham ===")
mono_ham = []

if os.path.exists("data/raw/normal.csv"):
    df = pd.read_csv("data/raw/normal.csv")
    tcol = next((c for c in df.columns
                 if 'text' in c.lower() or 'message' in c.lower()), df.columns[0])
    lcol = next((c for c in df.columns
                 if 'label' in c.lower() or 'spam' in c.lower() or 'class' in c.lower()), None)
    if lcol:
        sub = df[df[lcol].isin([0, 'ham', 'Ham', 'normal', 'legitimate'])]
        if len(sub) == 0: sub = df
    else:
        sub = df
    texts = sub[tcol].dropna().astype(str).tolist()
    mono_ham += texts
    print(f"  normal.csv: {len(texts)} rows")

if os.path.exists("data/raw/spam_ham_spanish.csv"):
    df = pd.read_csv("data/raw/spam_ham_spanish.csv")
    print(f"  spam_ham_spanish columns: {list(df.columns)[:5]}")
    tcol = next((c for c in df.columns
                 if 'text' in c.lower() or 'message' in c.lower() or 'sms' in c.lower()),
                df.columns[0])
    lcol = next((c for c in df.columns
                 if 'label' in c.lower() or 'spam' in c.lower() or 'class' in c.lower()), None)
    if lcol:
        print(f"    label dist: {df[lcol].value_counts().to_dict()}")
        sub = df[df[lcol].isin([0, 'ham', 'Ham', 'legitimate', 'normal'])]
        if len(sub) == 0: sub = df
    else:
        sub = df
    texts = sub[tcol].dropna().astype(str).tolist()
    mono_ham += texts
    print(f"  spam_ham_spanish.csv: {len(texts)} ham rows")

if os.path.exists("data/raw/scam_ham_india_14lang.csv"):
    df = pd.read_csv("data/raw/scam_ham_india_14lang.csv")
    print(f"  scam_ham_india_14lang columns: {list(df.columns)[:6]}")
    tcol = next((c for c in df.columns
                 if 'text' in c.lower() or 'message' in c.lower()), df.columns[0])
    lcol = next((c for c in df.columns
                 if 'label' in c.lower() or 'class' in c.lower() or 'spam' in c.lower()), None)
    if lcol:
        print(f"    label dist: {df[lcol].value_counts().to_dict()}")
        sub = df[df[lcol].isin([0, 'ham', 'Ham', 'legitimate', 'normal', 'Not Fraud'])]
        if len(sub) == 0: sub = df
    else:
        sub = df
    texts = sub[tcol].dropna().astype(str).tolist()
    mono_ham += texts
    print(f"  scam_ham_india_14lang.csv: {len(texts)} ham rows")

print(f"\n  Total mono ham pool: {len(mono_ham)}")
if len(mono_ham) > 5000:
    mono_ham = random.sample(mono_ham, 5000)
mono_ham_df = pd.DataFrame({
    'text':      mono_ham,
    'label':     0,
    'lang_pair': 'en',
    'category':  'mono_ham',
    'source':    'raw_ham',
})
for c in PERSUASION_COLS:
    mono_ham_df[c] = np.nan
other_parts.append(mono_ham_df)
print(f"  Using {len(mono_ham_df)} mono ham rows")

# ─── Assemble non-CS data ─────────────────────────────────────────────────────
print("\n=== Assembling and splitting non-CS data ===")
other = pd.concat(other_parts, ignore_index=True)
other['text'] = other['text'].astype(str).str.strip()
other = other[other['text'].str.len() > 10]
other = other.drop_duplicates(subset=['text'])

# Remove any texts already in CS splits to prevent leakage
cs_all_texts = set(cs_train['text'].tolist()) | set(cs_val['text'].tolist()) | set(cs_test['text'].tolist())
before_dedup = len(other)
other = other[~other['text'].isin(cs_all_texts)].reset_index(drop=True)
print(f"  Removed {before_dedup - len(other)} non-CS rows that overlapped with CS splits")
print(f"  Non-CS rows (before split): {len(other)}")

# Split non-CS 70/15/15 stratified by category
other_train, other_temp = train_test_split(
    other, test_size=0.30, random_state=42, stratify=other['category'])
other_val, other_test = train_test_split(
    other_temp, test_size=0.50, random_state=42, stratify=other_temp['category'])

# ─── Merge CS pre-split + non-CS split ────────────────────────────────────────
print("\n=== Merging CS phishing + non-CS splits ===")
train = pd.concat([cs_train, other_train], ignore_index=True).sample(
    frac=1, random_state=42).reset_index(drop=True)
val   = pd.concat([cs_val,   other_val],   ignore_index=True).sample(
    frac=1, random_state=42).reset_index(drop=True)
test  = pd.concat([cs_test,  other_test],  ignore_index=True).sample(
    frac=1, random_state=42).reset_index(drop=True)

full = pd.concat([train, val, test], ignore_index=True)

print(f"\nFull dataset: {len(full)} rows")
print("\nCategory breakdown:")
print(full.groupby(['category', 'label']).size().to_string())
print(f"\nLabel balance: {full['label'].value_counts().to_dict()}")
print(f"\nLang pair dist:\n{full['lang_pair'].value_counts().to_string()}")
print(f"\nPersuasion coverage: {full[PERSUASION_COLS[0]].notna().sum()} rows with persuasion labels")

full.to_csv(f"{OUT_DIR}/full_dataset.csv", index=False)
print(f"\nSaved → {OUT_DIR}/full_dataset.csv")

train.to_csv(f"{OUT_DIR}/train.csv", index=False)
val.to_csv(f"{OUT_DIR}/val.csv",     index=False)
test.to_csv(f"{OUT_DIR}/test.csv",   index=False)

print(f"\n  train: {len(train)}  val: {len(val)}  test: {len(test)}")
print(f"\n  Train label dist: {train['label'].value_counts().to_dict()}")
print(f"  Val   label dist: {val['label'].value_counts().to_dict()}")
print(f"  Test  label dist: {test['label'].value_counts().to_dict()}")

print("\nCS phishing per split:")
for split_name, df in [("train", train), ("val", val), ("test", test)]:
    n = (df['category'] == 'cs_phishing').sum()
    print(f"  {split_name}: {n} CS phishing rows")

print("\nDONE.")
