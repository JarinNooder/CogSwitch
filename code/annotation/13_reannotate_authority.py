"""
Script 13: Re-annotate ONLY the authority label with tightened definition.
Both Claude and GPT-4o re-annotate all 2794 CS phishing rows for authority only.

Tightened definition:
  authority=1 ONLY if the message explicitly IMPERSONATES an authority
  (claims to BE from a bank, government, brand, official institution).
  authority=0 if it merely MENTIONS a brand name without claiming to be from it.

Updates:
  data/annotated/cs_phishing_annotated_claude_full.csv  (persuasion_authority column)
  data/annotated/cs_phishing_annotated_gpt_full.csv     (persuasion_authority column)

Run from /home/nmunir/cogswitch with venv active:
  export ANTHROPIC_API_KEY="..."
  export OPENAI_API_KEY="..."
  python3 -u 13_reannotate_authority.py 2>&1 | tee logs/reannotate_authority.log
"""

import os, json, time
import pandas as pd
import anthropic
from openai import OpenAI

CLAUDE_MODEL = "claude-haiku-4-5-20251001"
GPT_MODEL    = "gpt-4o-2024-08-06"
IN_PATH      = "data/final_v2/full_dataset.csv"
CLAUDE_OUT   = "data/annotated/cs_phishing_annotated_claude_full.csv"
GPT_OUT      = "data/annotated/cs_phishing_annotated_gpt_full.csv"

SYSTEM_PROMPT = """You are an expert in social engineering and persuasion techniques.
You will be given a phishing/scam message (possibly in mixed languages).

Label whether the authority persuasion technique is present. Binary: 1=present, 0=absent.

Definition:
- authority=1: the message explicitly IMPERSONATES or claims to be FROM a trusted authority
  (e.g. "Your bank has sent you...", "Amazon: your order...", "Government tax notice",
  sender claims to be PayPal/Google/a bank/government agency).
- authority=0: the message merely MENTIONS a brand or organization without claiming to be from it,
  or contains no authority impersonation at all.

The key test: Is the message PRETENDING TO BE SENT BY an authority? If yes → 1. If not → 0.

Respond ONLY with valid JSON:
{"authority": 0_or_1}"""

claude_client = anthropic.Anthropic(api_key=os.environ["ANTHROPIC_API_KEY"])
gpt_client    = OpenAI()

def extract_json(raw):
    """Extract the first {...} block from a string."""
    start = raw.find('{')
    end   = raw.rfind('}')
    if start == -1 or end == -1:
        raise ValueError(f"No JSON object found in: {raw!r}")
    return json.loads(raw[start:end+1])

def annotate_claude_authority(text):
    try:
        resp = claude_client.messages.create(
            model=CLAUDE_MODEL,
            max_tokens=30,
            system=SYSTEM_PROMPT,
            messages=[{"role": "user", "content": f"Message: {text[:500]}"}],
        )
        raw = resp.content[0].text.strip()
        return int(bool(extract_json(raw).get("authority", 0)))
    except Exception as e:
        print(f"  Claude error: {e}")
        return None

def annotate_gpt_authority(text):
    try:
        resp = gpt_client.chat.completions.create(
            model=GPT_MODEL,
            messages=[
                {"role": "system", "content": SYSTEM_PROMPT},
                {"role": "user",   "content": f"Message: {text[:500]}"}
            ],
            temperature=0,
            max_tokens=20,
            response_format={"type": "json_object"},
        )
        return int(bool(json.loads(resp.choices[0].message.content).get("authority", 0)))
    except Exception as e:
        print(f"  GPT error: {e}")
        return None

# ── Load source rows ───────────────────────────────────────────────────────────
print(f"Loading {IN_PATH} ...")
df = pd.read_csv(IN_PATH)
df = df[(df['label'] == 1) & (df['lang_pair'] != 'en')].reset_index(drop=True)
print(f"Total CS phishing rows: {len(df)}")

# Load existing annotation files
claude_df = pd.read_csv(CLAUDE_OUT)
gpt_df    = pd.read_csv(GPT_OUT)

claude_results = {}  # orig_index -> new authority label
gpt_results    = {}

total  = len(df)
errors_c = 0
errors_g = 0

print("\nRe-annotating authority label...")
print(f"{'i':>6}  {'Claude':>8}  {'GPT':>6}  {'text[:60]'}")
print("-"*80)

for i, row in df.iterrows():
    text = str(row['text'])

    c = annotate_claude_authority(text)
    g = annotate_gpt_authority(text)

    if c is None:
        errors_c += 1
    else:
        claude_results[i] = c

    if g is None:
        errors_g += 1
    else:
        gpt_results[i] = g

    if (i+1) % 100 == 0:
        print(f"  [{i+1}/{total}] Claude errors: {errors_c}  GPT errors: {errors_g}")

    time.sleep(0.05)

print(f"\nDone. Claude errors: {errors_c}, GPT errors: {errors_g}")

# ── Update annotation files ────────────────────────────────────────────────────
# Claude
c_updated = 0
for idx, val in claude_results.items():
    mask = claude_df['orig_index'] == idx
    if mask.any():
        claude_df.loc[mask, 'persuasion_authority'] = val
        c_updated += 1
claude_df.to_csv(CLAUDE_OUT, index=False)
print(f"Claude: updated {c_updated} rows → {CLAUDE_OUT}")

# GPT
g_updated = 0
for idx, val in gpt_results.items():
    mask = gpt_df['orig_index'] == idx
    if mask.any():
        gpt_df.loc[mask, 'persuasion_authority'] = val
        g_updated += 1
gpt_df.to_csv(GPT_OUT, index=False)
print(f"GPT:    updated {g_updated} rows → {GPT_OUT}")

# ── Quick agreement check on authority only ────────────────────────────────────
from sklearn.metrics import cohen_kappa_score
import numpy as np

merged = claude_df[['orig_index','persuasion_authority']].merge(
    gpt_df[['orig_index','persuasion_authority']], on='orig_index', suffixes=('_c','_g')
).dropna()

c_vals = merged['persuasion_authority_c'].astype(int).tolist()
g_vals = merged['persuasion_authority_g'].astype(int).tolist()

kappa = cohen_kappa_score(c_vals, g_vals)
agree = sum(a==b for a,b in zip(c_vals,g_vals)) / len(c_vals) * 100
c_pct = sum(c_vals) / len(c_vals) * 100
g_pct = sum(g_vals) / len(g_vals) * 100

print(f"\nAuthority re-annotation results (N={len(merged)}):")
print(f"  Claude authority=1: {c_pct:.1f}%")
print(f"  GPT    authority=1: {g_pct:.1f}%")
print(f"  Agreement: {agree:.1f}%")
print(f"  Cohen's kappa: {kappa:.3f}")
print("\nNext: run 12b_kappa_full.py to get updated overall kappa.")
