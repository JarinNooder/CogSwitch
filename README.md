# CogSwitch: Supplementary Dataset and Reproducibility Materials

**Paper:** *CogSwitch: Detecting Code-Mixed Social Engineering Attacks via Cross-Lingual Supervised Contrastive Learning*

This archive accompanies an anonymized research submission. It provides the dataset splits, source code for the experimental pipeline, and saved explanation outputs. The materials are for research and reproducibility; no author-identifying information is included in this README.

## 1. Archive organization

Place this README at the root of the ZIP. The suggested organization is:

```text
CogSwitch/
├── README.md
├── code/
│   ├── train_cogswitch.py
│   ├── ablation_aux.py
│   ├── baselines.py
│   ├── fairness_eval_v2_fixed.py
│   ├── adversarial_robustness.py
│   ├── shap_explain_corrected.py
│   ├── lime_explain_revised.py
│   ├── tsne_viz_corrected.py
│   └── ... (additional supplied scripts)
├── data/
│   ├── train.csv
│   ├── val.csv
│   ├── test.csv
│   ├── full_dataset.csv
│   └── full_dataset_with_persuasion.csv
└── results/
    ├── shap_results.json
    └── lime_explanations.json
```

**Note:** This is a suggested layout, not an assertion about the ZIP's final structure. If you preserve the existing `data/final_v2/` directory or group scripts into subdirectories, retain those paths in the ZIP and adjust the commands below accordingly. In particular, most training/evaluation scripts expect `data/final_v2/` rather than `data/`.

## 2. Dataset

The principal dataset contains **19,698** records, partitioned as follows:

| File | Records | Purpose |
| --- | ---: | --- |
| `train.csv` | 13,786 | Model fitting |
| `val.csv` | 2,948 | Validation and checkpoint selection |
| `test.csv` | 2,964 | Held-out evaluation |
| `full_dataset.csv` | 19,698 | Combined principal dataset |
| `full_dataset_with_persuasion.csv` | 19,700 | Separate dataset export containing persuasion columns |

The first three splits sum to 19,698 rows. The `full_dataset_with_persuasion.csv` export contains **two additional monolingual phishing rows** relative to `full_dataset.csv`; do not treat the two full-dataset exports as interchangeable. The scripts for the reported final training/evaluation use the train/validation/test splits, not this alternate full-dataset export.

### Schema

All five CSVs contain:

- `text`: message text.
- `lang_pair`: language-pair identifier; the code-switched groups are `en_hi` (English–Hindi), `en_ar` (English–Arabic), `en_es` (English–Spanish), and `en_zh` (English–Chinese). `en` is the source/monolingual grouping label and should not be interpreted as a guarantee that every message is exclusively English.
- `label`: binary classification target (`0` = benign/ham; `1` = phishing).
- `category`: `cs_phishing`, `cs_ham`, `mono_phishing`, or `mono_ham`.
- `source`: data provenance/source tag.
- `persuasion_authority`, `persuasion_social_proof`, `persuasion_liking`, `persuasion_commitment`, `persuasion_distraction`: binary persuasion annotations where available; missing entries are not negative labels.

Persuasion labels are sparsely available. The training code masks missing persuasion annotations using `-1`. Do not fill missing labels with zero.

## 3. Method overview

CogSwitch uses an XLM-RoBERTa (`xlm-roberta-base`) encoder with a binary phishing classifier, a 128-dimensional projection head for supervised contrastive learning, and five auxiliary persuasion-classification heads. The training objective combines weighted binary cross-entropy, supervised contrastive loss, and masked auxiliary persuasion loss:

```text
L = (1 - lambda) * L_CE + lambda * L_SupCon + aux_weight * L_aux
```

The main reported setting uses `lambda = 0.7` and `aux_weight = 0.1`. Other lambda values are trained as **separate invocations of `train_cogswitch.py`**, not through a dedicated sweep script. The best checkpoint is selected using validation macro-F1.

Default training settings in `train_cogswitch.py` include 10 epochs, batch size 32, learning rate `2e-5`, maximum sequence length 128, contrastive temperature 0.1, and random seed 42.

## 4. Software and external dependencies

The scripts use Python 3 and, depending on the experiment, packages including:

```text
numpy
pandas
torch
transformers
scikit-learn
matplotlib
lime
shap
anthropic
openai
```

Dataset generation/validation scripts may additionally require FastText, a locally accessible Meta-Llama-3-8B-Instruct model, and their original source datasets. Persuasion annotation scripts require API credentials for the respective providers. Exact package versions and a fully pinned environment are not supplied here; differences in package versions, hardware, or model availability may affect reproducibility.

## 5. Training and evaluation

The following commands assume that scripts are accessible from the archive root and that the dataset has been placed at `data/final_v2/`. If your ZIP stores scripts under `code/`, invoke the corresponding path (for example, `python code/train_cogswitch.py`), while keeping the **working directory at the archive root**.

### Train the main model

```bash
python code/train_cogswitch.py --lam 0.7 --aux_weight 0.1 --data_dir data/final_v2
```

### Repeat the lambda experiments

```bash
python code/train_cogswitch.py --lam 0.0 --data_dir data/final_v2
python code/train_cogswitch.py --lam 0.3 --data_dir data/final_v2
python code/train_cogswitch.py --lam 0.5 --data_dir data/final_v2
python code/train_cogswitch.py --lam 0.7 --data_dir data/final_v2
python code/train_cogswitch.py --lam 0.9 --data_dir data/final_v2
```

Each run writes a best-checkpoint file and result history under `checkpoints/` by default. **Do not run the lambda 0.7 command twice unless you intend to overwrite its checkpoint.**

### Other experiments

| Script | Purpose | Important input or prerequisite |
| --- | --- | --- |
| `baselines.py` | TF-IDF classical baselines and mBERT | Training and test CSVs |
| `ablation_aux.py` | Compare auxiliary-loss weights at lambda 0.7 | Training, validation, and test CSVs |
| `fairness_eval_v2_fixed.py` | Per-language-pair error rates and equalized-odds differences, with bootstrap confidence intervals | Five saved lambda checkpoints |
| `adversarial_robustness.py` | Homoglyph and character-perturbation tests | Lambda 0.7 checkpoint and test CSV |
| `confusion_matrices.py` | Language-pair confusion matrices | Lambda 0.7 checkpoint and test CSV |
| `shap_explain_corrected.py` | Word-level SHAP analysis on 200 sampled test messages | Lambda 0.7 checkpoint and test CSV |
| `lime_explain_revised.py` | Illustrative LIME explanations for three language pairs | Lambda 0.7 checkpoint and test CSV |
| `tsne_viz_corrected.py` | Qualitative encoder-embedding visualization | Lambda 0.0 and 0.7 checkpoints |

**Path portability warning:** Some evaluation and explainability scripts contain absolute paths from the original compute environment (for example, `/home/nmunir/cogswitch/...`). These paths must be changed to the archive's local paths before execution. The README alone does not make those scripts directly runnable. Review all code, saved metadata, and generated logs for identifying paths before distributing a double-blind archive.

**Checkpoint availability:** Trained `.pt` checkpoints are not among the five dataset CSVs or two explanation JSONs described here. Experiments requiring checkpoints must first train them or obtain the corresponding checkpoint files separately. Reproducing the exact published results may require the original checkpoints and environment.

## 6. Saved explainability outputs

### `results/shap_results.json`

Saved aggregate word-level SHAP output from `shap_explain_corrected.py`, based on **200 fixed-seed test examples**. It includes the top tokens, their mean absolute SHAP attributions **conditional on token occurrence**, and occurrence counts (minimum two). These values should not be interpreted as causal evidence or as direct measurements of persuasion mechanisms.

### `results/lime_explanations.json`

Saved local LIME explanations for illustrative code-switched phishing messages from **English–Hindi, English–Arabic, and English–Spanish**. It records the selected texts, predicted phishing probabilities, token weights, local surrogate fidelity, sampling settings, and checkpoint metadata. English–Chinese is not included in the three-panel illustration.

**Interpretation caution:** The saved LIME explanations have very low local surrogate R² values (approximately 0.004–0.019) and small token weights. They are illustrative diagnostic outputs, **not reliable evidence of strong token-level causal effects**.

The LIME JSON also contains an original absolute checkpoint path. Remove or anonymize such paths in the JSON before distributing the ZIP; preserve the analytical values and clearly record any metadata-only sanitization.

## 7. Dataset construction and annotation provenance

Where included, the data-construction scripts document generation of code-mixed phishing messages using Llama 3, validation of generated messages, and assembly of the final splits. Relevant scripts include `08_generate_cs_v6_split.py`, `04e_validate_v6_splits.py`, and `10_assemble_dataset_v2_fixed.py`.

The generation process depends on source datasets and local model resources that are not necessarily bundled in this archive. Source-data licensing and redistribution conditions should be checked independently. Earlier generator versions, if retained, are provenance records and should not be mistaken for the final v6 pipeline.

The annotation scripts include Claude- and GPT-based persuasion labeling, inter-annotator agreement analysis, and an authority-label re-annotation procedure. Re-running API annotation may incur costs and can produce different outputs as hosted models change. The authority re-annotation script updates CSV files in place; work on copies rather than on the released dataset.

## 8. Reproducibility and responsible use

- The data contain phishing/social-engineering examples for defensive research. Do not use them to target real users.
- A fixed random seed does not guarantee bitwise identical results across GPU types, library versions, or external generative-model APIs.
- Language-pair labels are not demographic or nationality labels. Reported group differences should not be interpreted as individual-level fairness guarantees.
- Some generated messages may contain model artifacts, explanatory notes, or imperfect code-mixing. Inspect the examples and account for these limitations when interpreting results.
- Avoid publishing API keys, personal file paths, machine usernames, author metadata, or private URLs in a double-blind supplementary archive.

## 9. Citation

Please cite the accompanying anonymized CogSwitch manuscript when referring to these materials. Author information is intentionally omitted during double-blind review.
