import pandas as pd
import numpy as np
import random
import re
import os
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM

# ── config ────────────────────────────────────────────────────────────────────
RAW_DIR      = "data/raw"
OUT_DIR      = "data/generated_v6"
os.makedirs(OUT_DIR, exist_ok=True)

N_TARGET       = 1500
RANDOM_SEED    = 42
MAX_NEW_TOK    = 200
TEMPERATURE    = 0.9
BATCH_SIZE     = 8
INTENSITY_DIST = ["light"] * 20 + ["medium"] * 50 + ["heavy"] * 30
SPLIT_RATIOS   = {"train": 0.70, "val": 0.15, "test": 0.15}

# ── load model ────────────────────────────────────────────────────────────────
MODEL_ID   = os.path.expanduser("~/.cache/huggingface/hub/models--meta-llama--Meta-Llama-3-8B-Instruct/snapshots/")
snapshots  = sorted(os.listdir(MODEL_ID))
MODEL_PATH = MODEL_ID + snapshots[-1]
print(f"Loading model from {MODEL_PATH}")
tokenizer = AutoTokenizer.from_pretrained(MODEL_PATH, padding_side='left')
tokenizer.pad_token = tokenizer.eos_token
model = AutoModelForCausalLM.from_pretrained(
    MODEL_PATH, torch_dtype=torch.float16, device_map="auto"
)
model.eval()

# ── load English phishing pool ────────────────────────────────────────────────
df_eng   = pd.read_csv(f"{RAW_DIR}/phishing.csv")
eng_pool = df_eng["text"].dropna().tolist()
eng_pool = list(dict.fromkeys(eng_pool))   # deduplicate

random.seed(RANDOM_SEED)
np.random.seed(RANDOM_SEED)
indices = list(range(len(eng_pool)))
random.shuffle(indices)

n_total = len(indices)
n_train = int(n_total * SPLIT_RATIOS["train"])
n_val   = int(n_total * SPLIT_RATIOS["val"])
split_indices = {
    "train": indices[:n_train],
    "val":   indices[n_train:n_train + n_val],
    "test":  indices[n_train + n_val:],
}
split_pools = {s: [eng_pool[i] for i in idxs] for s, idxs in split_indices.items()}
print(f"Source pool: train={len(split_pools['train'])}, val={len(split_pools['val'])}, test={len(split_pools['test'])}")

# ── L2 pools ──────────────────────────────────────────────────────────────────
def load_l2(pair):
    keywords = {
        "en_hi": ["hinglish","hindi","scam"],
        "en_ar": ["arabic","arab"],
        "en_es": ["spanish","spam_ham","spaphish"],
        "en_zh": ["chinese","china"],
    }[pair]
    files = [f for f in os.listdir(RAW_DIR) if any(k in f.lower() for k in keywords)]
    out = []
    for f in files:
        try:
            d = pd.read_csv(f"{RAW_DIR}/{f}")
            if "text" in d.columns:
                out += d["text"].dropna().tolist()
            else:
                out += d.iloc[:,0].dropna().tolist()
        except Exception as e:
            print(f"  skipping {f}: {e}")
    return out

# ── prompt builder ────────────────────────────────────────────────────────────
INTENSITY_GUIDE = {
    "light":  "Mix in just a few words (10-25%) from {L2}. Most of the message stays in English.",
    "medium": "Mix roughly half (40-60%) of words from {L2} with English.",
    "heavy":  "Use {L2} for most of the message (65-80%), with English words sprinkled in.",
}
L2_NAMES = {"en_hi":"Hindi","en_ar":"Arabic","en_es":"Spanish","en_zh":"Mandarin"}

def build_prompt(eng_msg, l2_example, pair, intensity):
    L2    = L2_NAMES[pair]
    guide = INTENSITY_GUIDE[intensity].format(L2=L2)
    return (
        f"You are a bilingual speaker of English and {L2}. "
        f"Rewrite the following phishing SMS as a natural code-switching message. "
        f"{guide} "
        f"Keep the scam intent. Output ONLY the rewritten message, no explanation.\n\n"
        f"Original: {eng_msg}\n"
        f"Example {L2} style: {l2_example}\n"
        f"Rewritten:"
    )

# ── batch generation ──────────────────────────────────────────────────────────
def generate_batch(prompts):
    inputs = tokenizer(prompts, return_tensors="pt", padding=True,
                       truncation=True, max_length=1024).to(model.device)
    with torch.no_grad():
        out = model.generate(
            **inputs,
            max_new_tokens=MAX_NEW_TOK,
            temperature=TEMPERATURE,
            do_sample=True,
            pad_token_id=tokenizer.eos_token_id,
        )
    results = []
    for i, ids in enumerate(out):
        inp_len = inputs["input_ids"].shape[1]
        gen = tokenizer.decode(ids[inp_len:], skip_special_tokens=True).strip()
        gen = re.split(r'(?<=[.!?])\s+(?=[A-Z])', gen)[0]
        results.append(gen)
    return results

# ── main loop ─────────────────────────────────────────────────────────────────
eng_pool_index = {msg: i for i, msg in enumerate(eng_pool)}

PAIRS = ["en_hi", "en_ar", "en_es", "en_zh"]

for pair in PAIRS:
    l2_pool = load_l2(pair)
    print(f"\n=== {pair} | L2 pool: {len(l2_pool)} ===")
    if not l2_pool:
        print(f"  WARNING: no L2 pool, skipping")
        continue

    for split_name, pool in split_pools.items():
        n_target = max(30, int(N_TARGET * len(pool) / n_total))
        print(f"  {split_name}: pool={len(pool)}, target={n_target}")

        results      = []
        attempts     = 0
        max_attempts = n_target * 8
        intensity_queue = []

        while len(results) < n_target and attempts < max_attempts:
            # build a batch
            batch_prompts  = []
            batch_meta     = []
            for _ in range(BATCH_SIZE):
                if not intensity_queue:
                    intensity_queue = INTENSITY_DIST.copy()
                    random.shuffle(intensity_queue)
                intensity  = intensity_queue.pop()
                eng_msg    = random.choice(pool)
                l2_example = random.choice(l2_pool)
                batch_prompts.append(build_prompt(eng_msg, l2_example, pair, intensity))
                batch_meta.append((eng_msg, intensity))

            try:
                generated = generate_batch(batch_prompts)
            except Exception as e:
                print(f"  batch error: {e}")
                attempts += BATCH_SIZE
                continue

            for (eng_msg, intensity), text in zip(batch_meta, generated):
                attempts += 1
                if text and len(text.split()) >= 5:
                    results.append({
                        "text":      text,
                        "lang_pair": pair,
                        "label":     1,
                        "source":    "llama3_cs_v6",
                        "intensity": intensity,
                        "source_id": eng_pool_index.get(eng_msg, -1),
                        "eng_ref":   eng_msg,
                        "split":     split_name,
                    })
                if len(results) >= n_target:
                    break

            if attempts % 50 == 0:
                print(f"    attempts={attempts}, collected={len(results)}")

        out_path = f"{OUT_DIR}/{pair}_cs_v6_{split_name}.csv"
        pd.DataFrame(results).to_csv(out_path, index=False)
        print(f"  Saved {len(results)} rows -> {out_path}")

print("\nDone.")
