"""
Script 04e: Validate v6 split files using same logic as 04c.
Applies CS quality filter to each train/val/test split file independently.
- en_hi: romanized Hinglish keyword validator
- en_ar, en_es, en_zh: FastText LangID

Output: data/validated_v6/{pair}_cs_v6_{split}_validated.csv

Run from /home/nmunir/cogswitch with venv active.
"""
import os, re, warnings
import pandas as pd
import numpy as np
warnings.filterwarnings("ignore")

IN_DIR  = "data/generated_v6"
OUT_DIR = "data/validated_v6"
os.makedirs(OUT_DIR, exist_ok=True)

# ─── Romanized Hinglish validator ─────────────────────────────────────────────
ROMAN_HINDI = [
    'hai','hain','ka','ki','ke','ko','mein','se','aur','nahi','kya',
    'aapka','aapko','aapki','karein','karo','karna','hoga','hogi',
    'please','yahan','wahan','abhi','jaldi','lekin','matlab','sahi',
    'galat','paise','rupay','number','details','OTP','band','jayega',
    'account','bank','verify','click','link','update','urgent','turant'
]

def is_valid_hinglish(text):
    has_roman_hindi = any(w in text.lower().split() for w in ROMAN_HINDI)
    has_en = bool(re.search(r'[a-zA-Z]{3,}', text))
    return has_roman_hindi and has_en

# ─── FastText setup ────────────────────────────────────────────────────────────
print("Loading FastText LangID model...")
import fasttext
FT_MODEL_PATH = "models/lid.176.bin"
if not os.path.exists(FT_MODEL_PATH):
    print("  Downloading lid.176.bin...")
    import urllib.request
    os.makedirs("models", exist_ok=True)
    urllib.request.urlretrieve(
        "https://dl.fbaipublicfiles.com/fasttext/supervised-models/lid.176.bin",
        FT_MODEL_PATH)
ft = fasttext.load_model(FT_MODEL_PATH)
print("  FastText loaded.")

LANG2 = {"en_ar": "ar", "en_es": "es", "en_zh": "zh"}

def fasttext_cs_check(text, lang_pair):
    text = text.replace('\n', ' ').strip()
    preds = ft.predict(text, k=3)
    langs = [l.replace('__label__', '') for l in preds[0]]
    l2 = LANG2[lang_pair]
    return "en" in langs and l2 in langs

# ─── Validate each split file ─────────────────────────────────────────────────
summary = []

for pair in ["en_hi", "en_ar", "en_es", "en_zh"]:
    print(f"\n{'='*50}")
    print(f" {pair}")
    print(f"{'='*50}")
    pair_total_in = 0
    pair_total_out = 0

    for split in ["train", "val", "test"]:
        path = f"{IN_DIR}/{pair}_cs_v6_{split}.csv"
        if not os.path.exists(path):
            print(f"  MISSING: {path}")
            continue

        df = pd.read_csv(path)
        n_before = len(df)
        pair_total_in += n_before

        if pair == "en_hi":
            mask = df['text'].apply(is_valid_hinglish)
        else:
            mask = df['text'].apply(lambda t: fasttext_cs_check(t, pair))

        df_valid = df[mask].reset_index(drop=True)
        n_after = len(df_valid)
        pair_total_out += n_after

        out_path = f"{OUT_DIR}/{pair}_cs_v6_{split}_validated.csv"
        df_valid.to_csv(out_path, index=False)
        print(f"  {split}: {n_before} → {n_after} ({n_before-n_after} removed, {n_after/n_before*100:.1f}% kept)")
        summary.append((pair, split, n_before, n_after))

    print(f"  TOTAL {pair}: {pair_total_in} → {pair_total_out}")

print("\n" + "="*50)
print(" SUMMARY")
print("="*50)
print(f"{'Pair':<8} {'Split':<8} {'Before':>8} {'After':>8} {'Kept%':>7}")
print("-"*42)
for pair, split, nb, na in summary:
    print(f"{pair:<8} {split:<8} {nb:>8} {na:>8} {na/nb*100:>6.1f}%")

totals = {}
for pair, split, nb, na in summary:
    if pair not in totals:
        totals[pair] = [0, 0]
    totals[pair][0] += nb
    totals[pair][1] += na

print("\nPer-pair totals:")
grand_in = grand_out = 0
for pair, (nb, na) in totals.items():
    print(f"  {pair}: {nb} → {na} ({na/nb*100:.1f}% kept)")
    grand_in += nb
    grand_out += na
print(f"\nGrand total: {grand_in} → {grand_out} ({grand_out/grand_in*100:.1f}% kept)")
print("\nDONE. Validated files saved to data/validated_v6/")
