"""Reproducible multilingual LIME explanations for CogSwitch (lambda=0.7).

Run from the project environment: python3 lime_explain_revised.py
Requires: torch, transformers, lime, pandas, numpy, matplotlib.
Outputs: lime_explanations.png and lime_explanations.json.

Interpretation: LIME approximates changes in the model's *original* P(phishing)
under word deletion. Low fidelity or near-zero weights are reported, not hidden.
"""
import hashlib
import json
import os
import unicodedata
import warnings

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
from lime.lime_text import LimeTextExplainer
from transformers import AutoModel, AutoTokenizer

DATA_DIR = '/home/nmunir/cogswitch/data/final_v2'
CKPT_PATH = '/home/nmunir/cogswitch/checkpoints/best_lam0.7.pt'
OUT_PNG = '/home/nmunir/cogswitch/lime_explanations.png'
OUT_JSON = '/home/nmunir/cogswitch/lime_explanations.json'
MODEL_NAME = 'xlm-roberta-base'
MAX_LEN = 128
BATCH_SIZE = 32
NUM_FEATURES = 15
NUM_SAMPLES = 2000
SEED = 42
MIN_UNIQUE_WORDS = 15
CS_GROUPS = ('en_hi', 'en_ar', 'en_es')  # Three illustrative panels; en_zh intentionally excluded
DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


class CogSwitchModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.encoder = AutoModel.from_pretrained(MODEL_NAME)
        hidden = self.encoder.config.hidden_size
        self.projector = nn.Sequential(nn.Linear(hidden, hidden), nn.ReLU(), nn.Linear(hidden, 128))
        self.classifier = nn.Linear(hidden, 2)
        self.aux_heads = nn.ModuleList([nn.Linear(hidden, 2) for _ in range(5)])

    def forward(self, input_ids, attention_mask):
        cls = self.encoder(input_ids=input_ids, attention_mask=attention_mask).last_hidden_state[:, 0, :]
        return self.classifier(cls)


# Unicode-aware word splitting. Preserve combining marks inside Hindi/Arabic
# words; punctuation/whitespace are separators. LIME reconstructs the original
# text from the returned token offsets (bow=False).
def unicode_words(text):
    words = []
    start = None
    for i, char in enumerate(text):
        cat = unicodedata.category(char)
        is_word = cat[0] in ('L', 'N') or (cat[0] == 'M' and start is not None)
        if is_word:
            if start is None:
                start = i
        elif start is not None:
            words.append(text[start:i])
            start = None
    if start is not None:
        words.append(text[start:])
    return words


def load_model():
    if not os.path.isfile(CKPT_PATH):
        raise FileNotFoundError(CKPT_PATH)
    model = CogSwitchModel().to(DEVICE)
    try:
        checkpoint = torch.load(CKPT_PATH, map_location='cpu', weights_only=True)
    except TypeError:  # Older PyTorch
        checkpoint = torch.load(CKPT_PATH, map_location='cpu')
    state = checkpoint.get('model_state_dict', checkpoint)
    if all(k.startswith('module.') for k in state):
        state = {k[7:]: v for k, v in state.items()}
    model.load_state_dict(state, strict=True)
    model.eval()
    return model


def make_predictor(model, tokenizer):
    def predict_proba(texts):
        texts = list(texts)
        if not texts:
            return np.empty((0, 2), dtype=np.float64)
        chunks = []
        for i in range(0, len(texts), BATCH_SIZE):
            encoded = tokenizer(texts[i:i + BATCH_SIZE], truncation=True,
                                padding=True, max_length=MAX_LEN,
                                return_tensors='pt')
            with torch.inference_mode():
                logits = model(encoded['input_ids'].to(DEVICE),
                               encoded['attention_mask'].to(DEVICE))
                probs = torch.softmax(logits, dim=1).cpu().numpy()
            chunks.append(probs)
        return np.concatenate(chunks, axis=0)
    return predict_proba


def choose_example(frame, pair, pair_index, predict_proba):
    subset = frame[(frame['lang_pair'] == pair) & (frame['label'] == 1)].copy()
    subset = subset[subset['text'].map(lambda t: len(set(unicode_words(t))) >= MIN_UNIQUE_WORDS)]
    if subset.empty:
        raise RuntimeError(f'No phishing examples with {MIN_UNIQUE_WORDS} unique words for {pair}')
    # Fixed-seed ordering, no LIME-weight or confidence optimization.
    rng = np.random.default_rng(SEED + pair_index)
    ordered = subset.iloc[rng.permutation(len(subset))]
    for offset in range(0, len(ordered), BATCH_SIZE):
        batch = ordered.iloc[offset:offset + BATCH_SIZE]
        probs = predict_proba(batch['text'].tolist())[:, 1]
        good = np.flatnonzero(probs >= 0.5)
        if good.size:
            index = int(good[0])
            return batch.iloc[index], float(probs[index]), len(subset)
    raise RuntimeError(f'No correctly classified phishing example for {pair}')


def checkpoint_sha256(path):
    digest = hashlib.sha256()
    with open(path, 'rb') as f:
        for block in iter(lambda: f.read(1024 * 1024), b''):
            digest.update(block)
    return digest.hexdigest()


def main():
    print(f'Device: {DEVICE}')
    print(f'Checkpoint: {CKPT_PATH}')
    print(f'Checkpoint SHA256: {checkpoint_sha256(CKPT_PATH)}')
    model = load_model()
    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
    predict_proba = make_predictor(model, tokenizer)

    path = os.path.join(DATA_DIR, 'test.csv')
    frame = pd.read_csv(path).dropna(subset=['text', 'label', 'lang_pair']).copy()
    frame['text'] = frame['text'].astype(str)
    frame['label'] = frame['label'].astype(int)
    print(f'Test examples: {len(frame)}')
    selected = {}
    for i, pair in enumerate(CS_GROUPS):
        row, prob, pool_size = choose_example(frame, pair, i, predict_proba)
        seed = SEED + i
        explainer = LimeTextExplainer(
            class_names=['ham', 'phishing'],
            split_expression=unicode_words,
            bow=False,
            random_state=seed,
        )
        exp = explainer.explain_instance(
            row['text'], predict_proba,
            labels=(1,), num_features=NUM_FEATURES,
            num_samples=NUM_SAMPLES,
        )
        features = [(str(word), float(weight)) for word, weight in exp.as_list(label=1)]
        r2 = float(exp.score)
        selected[pair] = {
            'text': str(row['text']),
            'df_index': int(row.name),
            'confidence': prob,
            'seed_used': seed,
            'pool_size': pool_size,
            'local_fidelity_r2': r2,
            'local_surrogate_prediction': float(exp.local_pred[0]),
            'lime_features': features,
        }
        print(f'{pair}: index={row.name}, P(phishing)={prob:.6f}, '
              f'R2={r2:.3f}, top features={features[:3]}')
        if r2 < 0.5:
            warnings.warn(f'{pair}: low local surrogate fidelity (R2={r2:.3f}); interpret cautiously')
        if max((abs(weight) for _, weight in features), default=0) < 0.001:
            warnings.warn(f'{pair}: near-zero LIME weights; do not interpret as strong token effects')

    labels = {'en_hi': 'English–Hindi', 'en_ar': 'English–Arabic',
              'en_es': 'English–Spanish', 'en_zh': 'English–Chinese'}
    fig, axes = plt.subplots(1, 3, figsize=(18, 6.5))
    for ax, pair in zip(axes.flat, CS_GROUPS):
        info = selected[pair]
        features = sorted(info['lime_features'], key=lambda item: item[1])
        words = [word for word, _ in features]
        weights = [weight for _, weight in features]
        y = np.arange(len(words))
        ax.barh(y, weights,
                color=['#C16B6B' if weight > 0 else '#668EAC' for weight in weights],
                height=0.72)
        ax.set_yticks(y, labels=words, fontsize=9)
        ax.axvline(0, color='#666666', linewidth=0.9, linestyle='--')
        ax.set_title(f'{labels[pair]} ({pair})\nP(phishing)={info["confidence"]:.4f}; '
                     f'local R²={info["local_fidelity_r2"]:.2f}', fontsize=11)
        ax.set_xlabel('LIME weight (negative: ham; positive: phishing)', fontsize=9)
        ax.spines['top'].set_visible(False)
        ax.spines['right'].set_visible(False)
        ax.tick_params(axis='x', labelsize=8)
    fig.suptitle('CogSwitch (λ=0.7): illustrative multilingual LIME explanations\n'
                 f'Fixed-seed phishing examples; {NUM_SAMPLES:,} perturbations per example',
                 fontsize=13, fontweight='bold')
    fig.tight_layout(rect=(0, 0, 1, 0.88))
    fig.savefig(OUT_PNG, dpi=180, bbox_inches='tight')
    plt.close(fig)

    payload = {
        'checkpoint_path': CKPT_PATH,
        'checkpoint_sha256': checkpoint_sha256(CKPT_PATH),
        'num_samples_per_explanation': NUM_SAMPLES,
        'excluded_pair': 'en_zh',
        'exclusion_note': 'Excluded to retain the manuscript three-panel illustration; not evidence that all en_zh LIME explanations are degenerate.',
        'selection_rule': 'first correctly classified phishing message in fixed-seed permutation per pair, >=15 unique Unicode words',
        'tokenization': 'Unicode letters/numbers with combining marks kept within words',
        'notes': 'LIME weights describe local perturbation effects on unmodified P(phishing); local R2 measures surrogate fidelity.',
        'results': selected,
    }
    with open(OUT_JSON, 'w', encoding='utf-8') as f:
        json.dump(payload, f, ensure_ascii=False, indent=2)
    print(f'Saved: {OUT_PNG}\nSaved: {OUT_JSON}')


if __name__ == '__main__':
    main()
