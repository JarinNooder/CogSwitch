"""
Script 11b: Annotate ALL 2,794 CS phishing rows with Claude (claude-3-5-haiku).
Output: data/annotated/cs_phishing_annotated_claude_full.csv

This becomes the FINAL dataset labels.
GPT-4o annotations are used only for inter-annotator agreement (kappa/krippendorff)
on the overlapping rows.

Run from /home/nmunir/cogswitch with venv active:
  python3 -u 11b_annotate_claude_full.py 2>&1 | tee logs/annotate_claude_full.log
"""

import os, json, time
import pandas as pd
import anthropic

MODEL    = "claude-3-5-haiku-20241022"
IN_PATH  = "data/final_v2/full_dataset.csv"
OUT_PATH = "data/annotated/cs_phishing_annotated_claude_full.csv"
os.makedirs("data/annotated", exist_ok=True)
os.makedirs("logs", exist_ok=True)

PERSUASION_COLS = ['authority', 'social_proof', 'liking', 'commitment', 'distraction']

SYSTEM_PROMPT = """You are an expert in social engineering and persuasion techniques.
You will be given a phishing/scam message (possibly in mixed languages — Hinglish, Arabizi, Spanglish, or Chinese-English).
Label which of the 5 persuasion techniques are present. Each is binary: 1=present, 0=absent.

Definitions:
- authority: message impersonates or invokes a trusted authority (bank, government, brand like Amazon/Google/PayPal, official institution)
- social_proof: references what others are doing or have done ("many users", "customers like you", "your friends")
- liking: uses flattery, personalization, or familiarity to seem friendly/trustworthy ("Dear valued customer", uses recipient's name, overly friendly tone)
- commitment: exploits a prior commitment or ownership ("your account", "your order", "your package") to make the message feel personal and legitimate
- distraction: creates urgency, fear, or time pressure to prevent the recipient from thinking carefully ("immediately", "within 24 hours", "or your account will be closed", "URGENT")

Respond ONLY with valid JSON, no explanation:
{"authority": 0_or_1, "social_proof": 0_or_1, "liking": 0_or_1, "commitment": 0_or_1, "distraction": 0_or_1}"""

client = anthropic.Anthropic(api_key=os.environ["ANTHROPIC_API_KEY"])

def annotate_claude(text):
    """Annotate a single message. Returns dict or None on error."""
    try:
        resp = client.messages.create(
            model=MODEL,
            max_tokens=60,
            system=SYSTEM_PROMPT,
            messages=[{"role": "user", "content": f"Message: {text[:500]}"}],
        )
        result = json.loads(resp.content[0].text)
        for k in PERSUASION_COLS:
            result[k] = int(bool(result.get(k, 0)))
        return result
    except Exception as e:
        print(f"  Error: {e}")
        return None

# ── Load source data ───────────────────────────────────────────────────────────
print(f"Loading {IN_PATH} ...")
df = pd.read_csv(IN_PATH)

# Keep only CS phishing rows (label=1, non-English)
df = df[(df['label'] == 1) & (df['lang_pair'] != 'en')].reset_index(drop=True)
print(f"Total CS phishing rows to annotate: {len(df)}")
print(f"By lang_pair:\n{df['lang_pair'].value_counts().to_string()}")

# Resume support: load already-done rows
done_ids  = set()
done_rows = []
if os.path.exists(OUT_PATH):
    done_df  = pd.read_csv(OUT_PATH)
    done_ids = set(done_df['orig_index'].tolist()) if 'orig_index' in done_df.columns else set()
    done_rows = done_df.to_dict('records')
    print(f"Resuming: {len(done_rows)} already annotated")

# Load GPT-4o annotations for overlap (kappa later)
gpt_lookup = {}
gpt_path = "data/annotated/cs_phishing_annotated.csv"
if os.path.exists(gpt_path):
    gpt_df = pd.read_csv(gpt_path)
    for _, r in gpt_df.iterrows():
        gpt_lookup[str(r['text']).strip()] = r
    print(f"Loaded {len(gpt_lookup)} GPT-4o annotations for overlap")

# ── Annotation loop ────────────────────────────────────────────────────────────
errors = 0
total  = len(df)

for i, row in df.iterrows():
    if i in done_ids:
        continue

    text = str(row['text'])
    result = annotate_claude(text)

    if result is None:
        errors += 1
        print(f"  [{i}/{total}] ERROR (total errors: {errors})")
        if errors > 20:
            print("Too many consecutive errors — stopping. Re-run to resume.")
            break
        time.sleep(3)
        continue

    errors = 0  # reset on success

    record = {
        'orig_index': i,
        'text':       text,
        'lang_pair':  row.get('lang_pair', ''),
        # Claude labels (FINAL dataset labels)
        'persuasion_authority':    result['authority'],
        'persuasion_social_proof': result['social_proof'],
        'persuasion_liking':       result['liking'],
        'persuasion_commitment':   result['commitment'],
        'persuasion_distraction':  result['distraction'],
        'annotator': MODEL,
    }

    # Carry GPT-4o labels if this text exists in GPT annotations (for kappa)
    gpt_row = gpt_lookup.get(text.strip())
    if gpt_row is not None:
        for col in ['persuasion_authority', 'persuasion_social_proof',
                    'persuasion_liking', 'persuasion_commitment', 'persuasion_distraction']:
            record[f'gpt_{col}'] = gpt_row.get(col, None)

    done_rows.append(record)
    done_ids.add(i)

    n_done = len(done_rows)
    if n_done % 50 == 0:
        pd.DataFrame(done_rows).to_csv(OUT_PATH, index=False)
        pct = n_done / total * 100
        print(f"  [{n_done}/{total}] {pct:.1f}% done — saved checkpoint")

    time.sleep(0.05)  # ~1200 RPM haiku limit

# Final save
out_df = pd.DataFrame(done_rows)
out_df.to_csv(OUT_PATH, index=False)
print(f"\nDone. {len(out_df)}/{total} rows saved → {OUT_PATH}")

# ── Quick stats ────────────────────────────────────────────────────────────────
print("\n" + "="*55)
print("ANNOTATION SUMMARY")
print("="*55)
print(f"Total annotated: {len(out_df)}")
print(f"\nLabel prevalence (Claude):")
for col in ['persuasion_authority', 'persuasion_social_proof',
            'persuasion_liking', 'persuasion_commitment', 'persuasion_distraction']:
    pct = out_df[col].mean() * 100
    label = col.replace('persuasion_', '')
    print(f"  {label:<20} {pct:5.1f}%")

print(f"\nBy lang_pair:")
print(out_df['lang_pair'].value_counts().to_string())

gpt_cols = [c for c in out_df.columns if c.startswith('gpt_persuasion')]
if gpt_cols:
    overlap = out_df[gpt_cols[0]].notna().sum()
    print(f"\nRows with GPT-4o labels (overlap for kappa): {overlap}")

print("\nNext step: run 12b_kappa_full.py to compute kappa/krippendorff on overlapping rows.")
