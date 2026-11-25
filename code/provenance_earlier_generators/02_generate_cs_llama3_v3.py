"""
Script 02 v3: Code-Switching generation using Llama-3 8B Instruct.
Loads model locally from models/llama3-8b-instruct/
Generates CS phishing messages for all 4 language pairs.
"""
import os, re, json, random
import pandas as pd
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM

# ─── Config ───────────────────────────────────────────────────────────────────
MODEL_PATH   = "models/llama3-8b-instruct"
DATA_DIR     = "data/raw"
OUT_DIR      = "data/cs_generated"
N_PER_PAIR   = 300   # generate 300 per pair, filter down to ~200
SEED         = 42
random.seed(SEED)
os.makedirs(OUT_DIR, exist_ok=True)

# ─── Load model ───────────────────────────────────────────────────────────────
print("Loading Llama-3 8B Instruct...")
tokenizer = AutoTokenizer.from_pretrained(MODEL_PATH)
model = AutoModelForCausalLM.from_pretrained(
    MODEL_PATH,
    torch_dtype=torch.float16,
    device_map="auto"
)
model.eval()
print("Model loaded.")

def generate(prompt, max_new_tokens=200, temperature=0.85, top_p=0.92):
    inputs = tokenizer(prompt, return_tensors="pt").to(model.device)
    with torch.no_grad():
        out = model.generate(
            **inputs,
            max_new_tokens=max_new_tokens,
            temperature=temperature,
            top_p=top_p,
            do_sample=True,
            pad_token_id=tokenizer.eos_token_id
        )
    decoded = tokenizer.decode(out[0], skip_special_tokens=True)
    # Return only the generated part after the prompt
    return decoded[len(tokenizer.decode(inputs["input_ids"][0], skip_special_tokens=True)):].strip()

def make_prompt(lang_pair, en_msg, l2_msg):
    """Build a few-shot prompt for CS mixing."""
    instructions = {
        "en_hi": (
            "You are creating Hinglish phishing messages. "
            "Hinglish mixes Hindi (Devanagari or romanized Hindi) and English naturally in one sentence. "
            "Use realistic Indian phishing patterns: bank KYC, OTP fraud, government schemes, parcel scams."
        ),
        "en_ar": (
            "You are creating Arabic-English code-switched phishing messages. "
            "Mix Arabic words/phrases with English naturally. "
            "Use realistic patterns: bank account suspension, prize winning, customs fees."
        ),
        "en_es": (
            "You are creating Spanish-English code-switched phishing messages (Spanglish). "
            "Mix Spanish and English naturally in one message. "
            "Use realistic patterns: account suspension, prize notification, delivery fees."
        ),
        "en_zh": (
            "You are creating Chinese-English code-switched phishing messages. "
            "Mix Chinese characters with English words/phrases naturally. "
            "Use realistic patterns: fake card services, loan fraud, account verification."
        ),
    }

    examples = {
        "en_hi": [
            "Alert: Aapka SBI account 24 ghante mein block ho jayega. KYC update karne ke liye abhi click karein: sbi-kyc.in",
            "Congratulations! Aapko ₹50,000 ka prize mila hai. Claim karne ke liye OTP share karein: 7894561230",
        ],
        "en_ar": [
            "تنبيه: Your account has been suspended. لإعادة تفعيل حسابك، click here: bank-verify.com",
            "مبروك! You have won $500 prize. لاستلام الجائزة، provide your card details now.",
        ],
        "en_es": [
            "ALERTA: Su cuenta ha sido suspended. Click here to verify your identity immediately: banco-seguro.com",
            "¡Felicidades! You have been selected for a special offer. Llame ahora: 1-800-555-0123",
        ],
        "en_zh": [
            "您的账户 has been suspended. Please verify immediately: bank-secure.cn to avoid 账户冻结.",
            "恭喜您! You have won ¥10,000. 请立即联系 our agent at: prize-claim.cn",
        ],
    }

    ex = examples[lang_pair]
    inst = instructions[lang_pair]

    prompt = f"""<|begin_of_text|><|start_header_id|>system<|end_header_id|>
{inst}
Generate a single realistic code-switched phishing message. Output ONLY the message text, nothing else.<|eot_id|>
<|start_header_id|>user<|end_header_id|>
Here are example messages for style reference:
- {ex[0]}
- {ex[1]}

Now create a NEW code-switched phishing message that combines elements from these two source messages:
English phishing: "{en_msg}"
L2 phishing: "{l2_msg}"

Generate one natural code-switched message:<|eot_id|>
<|start_header_id|>assistant<|end_header_id|>
"""
    return prompt

def is_valid_cs(text, lang_pair):
    """Check that generated text actually contains both languages."""
    if len(text) < 20 or len(text) > 500:
        return False
    # Must have some English (ASCII letters)
    english_ratio = len(re.findall(r'[a-zA-Z]', text)) / max(len(text), 1)
    if english_ratio < 0.05:
        return False

    if lang_pair == "en_hi":
        # Must have Devanagari or romanized Hindi markers
        has_devanagari = bool(re.search(r'[ऀ-ॿ]', text))
        hindi_markers = ['aapka','aapko','karein','karo','hai','hoga','mein','ka','ki','ke']
        has_markers = any(m in text.lower() for m in hindi_markers)
        return has_devanagari or (has_markers and english_ratio < 0.85)

    elif lang_pair == "en_ar":
        has_arabic = bool(re.search(r'[؀-ۿ]', text))
        return has_arabic and english_ratio < 0.90

    elif lang_pair == "en_es":
        spanish_markers = ['su','cuenta','haga','clic','aquí','ahora','usted','para','con','una','por']
        has_spanish = any(m in text.lower() for m in spanish_markers)
        return has_spanish and 0.05 < english_ratio < 0.95

    elif lang_pair == "en_zh":
        has_chinese = bool(re.search(r'[一-鿿]', text))
        return has_chinese and english_ratio < 0.90

    return False

# ─── Data loading ─────────────────────────────────────────────────────────────
def load_english_phishing(n=200):
    df = pd.read_csv(f"{DATA_DIR}/phishing.csv")
    text_col = next((c for c in ["text","body","message","Email Text"] if c in df.columns), df.columns[0])
    return df[text_col].dropna().sample(min(n, len(df)), random_state=SEED).tolist()

# ─── Generate per pair ────────────────────────────────────────────────────────
pairs = {
    "en_hi": {
        "l2_file": f"{DATA_DIR}/scam_ham_india_14lang.csv",
        "l2_text_col": "text",
        "l2_filter": lambda df: df[df["language"] == "Hindi"][df["label"] == 1],
    },
    "en_ar": {
        "l2_file": f"{DATA_DIR}/arabic_spam_tweets.csv",
        "l2_text_col": "text",
        "l2_filter": lambda df: df,
    },
    "en_es": {
        "l2_file": f"{DATA_DIR}/spaphish_spanish.csv",
        "l2_text_col": "text",
        "l2_filter": lambda df: df[df["label"] == 1],
    },
    "en_zh": {
        "l2_file": f"{DATA_DIR}/chifraud_chinese.csv",
        "l2_text_col": "text",
        "l2_filter": lambda df: df[df["label"] == 1],
    },
}

en_msgs = load_english_phishing(N_PER_PAIR)
print(f"Loaded {len(en_msgs)} English phishing messages")

for pair, cfg in pairs.items():
    print(f"\n{'='*60}")
    print(f" Generating {pair} ({N_PER_PAIR} attempts)")
    print('='*60)

    # Load L2 phishing
    l2_df = pd.read_csv(cfg["l2_file"])
    l2_df = cfg["l2_filter"](l2_df)
    l2_msgs = l2_df[cfg["l2_text_col"]].dropna().sample(
        min(N_PER_PAIR, len(l2_df)), random_state=SEED).tolist()
    print(f"L2 source: {len(l2_msgs)} messages from {cfg['l2_file']}")

    results = []
    for i in range(min(N_PER_PAIR, len(l2_msgs))):
        en_msg = random.choice(en_msgs)[:300]
        l2_msg = l2_msgs[i][:300]
        prompt = make_prompt(pair, en_msg, l2_msg)

        try:
            generated = generate(prompt)
            # Take only first line if multi-line
            generated = generated.split('\n')[0].strip()
            generated = re.sub(r'^[-•*]\s*', '', generated)

            valid = is_valid_cs(generated, pair)
            if valid:
                results.append({
                    "text": generated,
                    "lang_pair": pair,
                    "label": 1,
                    "source": "llama3_cs_generated"
                })

            if (i+1) % 20 == 0:
                print(f"  [{i+1}/{N_PER_PAIR}] Valid so far: {len(results)}")
                if results:
                    print(f"  Latest: {results[-1]['text'][:100]}")

        except Exception as e:
            print(f"  Error at {i}: {e}")
            continue

    print(f"\n✅ {pair}: {len(results)} valid CS messages generated")
    if results:
        out_df = pd.DataFrame(results)
        out_path = f"{OUT_DIR}/{pair}_cs_generated.csv"
        out_df.to_csv(out_path, index=False)
        print(f"   Saved → {out_path}")
    else:
        print(f"   ⚠️ No valid messages — check prompts and filters")

print("\n\nDONE. Summary:")
for pair in pairs:
    path = f"{OUT_DIR}/{pair}_cs_generated.csv"
    if os.path.exists(path):
        df = pd.read_csv(path)
        print(f"  {pair}: {len(df)} rows → {path}")
    else:
        print(f"  {pair}: NOT GENERATED")
