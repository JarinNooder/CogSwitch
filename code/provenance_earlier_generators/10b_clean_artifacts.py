"""
Script 10b: Clean Llama-3 generation artifacts from text fields.
Strips meta-notes appended by Llama-3 during CS generation.

NO re-splitting, NO label changes, NO row removal.
Only the text field is cleaned in-place.

Cleans:
  - data/final_v2/full_dataset.csv
  - data/final_v2/train.csv
  - data/final_v2/val.csv
  - data/final_v2/test.csv

Run from /home/nmunir/cogswitch with venv active:
  python3 -u 10b_clean_artifacts.py
"""

import re
import pandas as pd
import os

# All artifact marker patterns — strip from marker to end of string
MARKERS = [
    r'\n\nNote:',
    r'\nNote:',
    r'\n\n\(Note',
    r'\n\(Note',
    r'\s*\(Note:',          # inline (Note: ...) anywhere
    r'\s+Note:\s+"',        # inline Note: "..." at end
    r'\s*-\s*Note:',        # - Note: pattern
    r'\s+Note:',            # space + Note: at end
    r'[。！，.!,]\s*Note:', # after CJK/full-width punctuation
    r'[。！，.!,]\s*\(Note:',
    r'[！，!,]\s*（Note:',  # full-width （Note:
    r'\n\nTranslation:',
    r'\nTranslation:',
    r'\n\n\(Translation',
    r'\n\(Translation',
    r'\s*\(Translation:',   # inline (Translation: ...) anywhere
    r'\n\nPlease note:',
    r'\nPlease note:',
    r'\s*\(Please note:',   # inline (Please note: ...) anywhere
    r'\n\nCode-switching',
    r'\nCode-switching',
    r'\n\nExample:',
    r'\nExample:',
    r'\[Note:',
    r'\[Translation:',
    r'\n\n\n?Hint:',        # Hint: at end
    r'\s*Hint:',            # inline Hint: anywhere
]

# Build one pattern: match any marker then everything to end
PATTERN = re.compile(
    '(' + '|'.join(MARKERS) + ')' + r'.*',
    re.DOTALL | re.IGNORECASE
)

def clean_text(text):
    if not isinstance(text, str):
        return text
    return PATTERN.sub('', text).strip()

def clean_file(path):
    print(f"\nCleaning {path} ...")
    df = pd.read_csv(path)
    n = len(df)

    # Backup original (only once)
    bak = path.replace('.csv', '.bak.csv')
    if not os.path.exists(bak):
        df.to_csv(bak, index=False)
        print(f"  Backup saved → {bak}")

    before = df['text'].copy()
    df['text'] = df['text'].apply(clean_text)
    changed = (df['text'] != before).sum()
    print(f"  Rows cleaned: {changed}/{n}")

    assert len(df) == n, "ERROR: row count changed!"

    # Check remaining
    for marker in ['Note:', 'Translation:', 'Code-switching', 'Please note:']:
        remaining = df['text'].str.contains(marker, na=False, regex=False).sum()
        if remaining > 0:
            print(f"  WARNING: {remaining} rows still contain '{marker}'")

    df.to_csv(path, index=False)
    print(f"  Saved → {path}")
    return changed

# ── Clean all final_v2 files ───────────────────────────────────────────────────
files = [
    "data/final_v2/full_dataset.csv",
    "data/final_v2/train.csv",
    "data/final_v2/val.csv",
    "data/final_v2/test.csv",
]

total_changed = 0
for f in files:
    if os.path.exists(f):
        total_changed += clean_file(f)
    else:
        print(f"MISSING: {f}")

print(f"\nTotal rows cleaned across all files: {total_changed}")
print("\nNext steps:")
print("  1. rm data/annotated/cs_phishing_annotated_claude_full.csv")
print("  2. python3 -u 11b_annotate_claude_full.py 2>&1 | tee logs/annotate_claude_full.log")
print("  3. python3 -u 12b_kappa_full.py 2>&1 | tee logs/kappa_full.log")
print("\nDONE.")
