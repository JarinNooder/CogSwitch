"""
Script 12b: Inter-Annotator Agreement — full overlap (all 2,794 rows).
Computes Cohen's Kappa AND Krippendorff's Alpha (GPT-4o vs Claude)
on all rows annotated by both annotators.

Inputs:
  data/annotated/cs_phishing_annotated_claude_full.csv  (2779 rows)
  data/annotated/cs_phishing_annotated_gpt_full.csv     (2794 rows)

Run from /home/nmunir/cogswitch with venv active:
  python3 -u 12b_kappa_full.py 2>&1 | tee logs/kappa_full.log
"""

import os
import pandas as pd
import numpy as np
from sklearn.metrics import cohen_kappa_score

# ── Krippendorff's Alpha (binary nominal) ─────────────────────────────────────
def krippendorff_alpha_binary(a1, a2):
    a1 = np.array(a1)
    a2 = np.array(a2)
    n  = len(a1)
    if n < 2:
        return float('nan')
    Do = np.sum(a1 != a2) / n
    all_vals = np.concatenate([a1, a2])
    n_total  = len(all_vals)
    vals, counts = np.unique(all_vals, return_counts=True)
    freq = dict(zip(vals, counts))
    De = 0.0
    for v1 in vals:
        for v2 in vals:
            if v1 != v2:
                De += (freq.get(v1, 0) / n_total) * (freq.get(v2, 0) / n_total)
    if De == 0:
        return 1.0
    return 1.0 - (Do / De)

# ── Load data ──────────────────────────────────────────────────────────────────
CLAUDE_PATH = "data/annotated/cs_phishing_annotated_claude_full.csv"
GPT_PATH    = "data/annotated/cs_phishing_annotated_gpt_full.csv"

print(f"Loading {CLAUDE_PATH} ...")
claude_df = pd.read_csv(CLAUDE_PATH)
print(f"Loading {GPT_PATH} ...")
gpt_df    = pd.read_csv(GPT_PATH)

print(f"Claude rows: {len(claude_df)}")
print(f"GPT-4o rows: {len(gpt_df)}")

# Merge on orig_index (inner join = rows both annotated)
claude_df = claude_df.rename(columns={
    'persuasion_authority':    'c_authority',
    'persuasion_social_proof': 'c_social_proof',
    'persuasion_liking':       'c_liking',
    'persuasion_commitment':   'c_commitment',
    'persuasion_distraction':  'c_distraction',
})[['orig_index', 'lang_pair',
    'c_authority', 'c_social_proof', 'c_liking', 'c_commitment', 'c_distraction']]

gpt_df = gpt_df.rename(columns={
    'persuasion_authority':    'g_authority',
    'persuasion_social_proof': 'g_social_proof',
    'persuasion_liking':       'g_liking',
    'persuasion_commitment':   'g_commitment',
    'persuasion_distraction':  'g_distraction',
})[['orig_index',
    'g_authority', 'g_social_proof', 'g_liking', 'g_commitment', 'g_distraction']]

overlap = claude_df.merge(gpt_df, on='orig_index', how='inner')
print(f"\nOverlap rows (annotated by both): {len(overlap)}")
print(f"By lang_pair:\n{overlap['lang_pair'].value_counts().to_string()}\n")

LABELS = ['authority', 'social_proof', 'liking', 'commitment', 'distraction']

# ── Per-label stats ────────────────────────────────────────────────────────────
print("="*75)
print("INTER-ANNOTATOR AGREEMENT: GPT-4o vs Claude")
print(f"N = {len(overlap)} rows (full overlap)")
print("="*75)

header = f"{'Label':<20} {'κ':>8}  {'α':>8}  {'GPT%':>7}  {'Claude%':>8}  {'Agree%':>8}"
print(header)
print("-"*75)

kappas = {}
alphas = {}

for label in LABELS:
    g = overlap[f'g_{label}'].astype(int).tolist()
    c = overlap[f'c_{label}'].astype(int).tolist()

    kappa = cohen_kappa_score(g, c)
    alpha = krippendorff_alpha_binary(g, c)
    kappas[label] = kappa
    alphas[label]  = alpha

    gpt_pct    = sum(g) / len(g) * 100
    claude_pct = sum(c) / len(c) * 100
    agree_pct  = sum(gi == ci for gi, ci in zip(g, c)) / len(g) * 100

    print(f"{label:<20} {kappa:>8.3f}  {alpha:>8.3f}  {gpt_pct:>6.1f}%  {claude_pct:>7.1f}%  {agree_pct:>7.1f}%")

avg_kappa = sum(kappas.values()) / len(kappas)
avg_alpha = sum(alphas.values()) / len(alphas)
print("-"*75)
print(f"{'AVERAGE':<20} {avg_kappa:>8.3f}  {avg_alpha:>8.3f}")

# ── Interpretation ─────────────────────────────────────────────────────────────
def interpret_kappa(k):
    if k < 0:    return "Poor"
    if k < 0.20: return "Slight"
    if k < 0.40: return "Fair"
    if k < 0.60: return "Moderate"
    if k < 0.80: return "Substantial"
    return               "Almost Perfect"

def interpret_alpha(a):
    if a < 0:    return "No agreement"
    if a < 0.20: return "Very weak"
    if a < 0.40: return "Weak"
    if a < 0.60: return "Moderate"
    if a < 0.67: return "Tentative (just below threshold)"
    if a < 0.80: return "Acceptable (>=0.667)"
    return               "Strong (>=0.800)"

print("\nInterpretation (Landis & Koch 1977 for kappa; Krippendorff 2004 for alpha):")
print(f"  {'Label':<20} {'kappa interpret':<22}  {'alpha interpret'}")
print("  " + "-"*65)
for label in LABELS:
    k = kappas[label]; a = alphas[label]
    print(f"  {label:<20} {interpret_kappa(k):<22}  {interpret_alpha(a)}")
print(f"  {'AVERAGE':<20} {interpret_kappa(avg_kappa):<22}  {interpret_alpha(avg_alpha)}")

# ── Per-language-pair breakdown ────────────────────────────────────────────────
print("\nPer language pair:")
print(f"  {'Pair':<12} {'N':>5}  {'avg kappa':>10}  {'avg alpha':>10}")
print("  " + "-"*42)
for pair in sorted(overlap['lang_pair'].unique()):
    sub = overlap[overlap['lang_pair'] == pair]
    pair_kappas, pair_alphas = [], []
    for label in LABELS:
        g = sub[f'g_{label}'].astype(int).tolist()
        c = sub[f'c_{label}'].astype(int).tolist()
        try:
            pair_kappas.append(cohen_kappa_score(g, c))
            pair_alphas.append(krippendorff_alpha_binary(g, c))
        except Exception:
            pass
    avg_k = sum(pair_kappas) / len(pair_kappas) if pair_kappas else float('nan')
    avg_a = sum(pair_alphas) / len(pair_alphas) if pair_alphas else float('nan')
    print(f"  {pair:<12} {len(sub):>5}  {avg_k:>10.3f}  {avg_a:>10.3f}")

print("\n" + "="*75)
print("SUMMARY FOR PAPER")
print("="*75)
print(f"  Annotator 1: GPT-4o (gpt-4o-2024-08-06), N={len(gpt_df)} rows")
print(f"  Annotator 2: Claude (claude-haiku-4-5), N={len(claude_df)} rows (used as final labels)")
print(f"  Overlap for IAA: {len(overlap)} rows")
print(f"  Average Cohen's kappa  : {avg_kappa:.3f}  ({interpret_kappa(avg_kappa)})")
print(f"  Average Krippendorff alpha: {avg_alpha:.3f}  ({interpret_alpha(avg_alpha)})")
print("\nDone.")
