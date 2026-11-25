# CogSwitch: Code-Mixed Social Engineering Attack Detection

**Paper:** *CogSwitch: Detecting Code-Mixed Social Engineering Attacks via Cross-Lingual Supervised Contrastive Learning*

This repository provides source code, dataset splits, and saved analysis outputs accompanying the CogSwitch research paper. CogSwitch uses XLM-RoBERTa, supervised contrastive learning, and auxiliary persuasion classification to study phishing detection in code-mixed multilingual messages. The materials are shared to support research transparency and reproducibility, subject to the restrictions noted below.

## 1. Repository organization

The public repository is organized as follows (individual files and nested paths should be checked against the current repository):

```
CogSwitch/
├── README.md
├── requirements.txt
├── CITATION.cff
├── LICENSE
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
├── dataset/
│   ├── train.csv
│   ├── val.csv
│   ├── test.csv
│   ├── full_dataset.csv
│   └── full_dataset_with_persuasion.csv
└── results/
    ├── shap_results.json
    └── lime_explanations.json

```

**Path note:** The repository uses `dataset/` at the top level. Confirm whether the CSV files are immediately inside `dataset/` or in a subdirectory before using the example commands. Several original scripts may still use the former `data/final_v2/` location or hard-coded server paths.

## 2. Dataset

The principal dataset contains **19,698** records, partitioned as follows:

| **File**                           | **Records** | **Purpose**                                           |
| ---------------------------------- | ----------- | ----------------------------------------------------- |
| `train.csv`                        | 13,786      | Model fitting                                         |
| `val.csv`                          | 2,948       | Validation and checkpoint selection                   |
| `test.csv`                         | 2,964       | Held-out evaluation                                   |
| `full_dataset.csv`                 | 19,698      | Combined principal dataset                            |
| `full_dataset_with_persuasion.csv` | 19,700      | Separate dataset export containing persuasion columns |

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

```
L = (1 - lambda) * L_CE + lambda * L_SupCon + aux_weight * L_aux

```

The main reported setting uses `lambda = 0.7` and `aux_weight = 0.1`. Other lambda values are trained as **separate invocations of `train_cogswitch.py`**, not through a dedicated sweep script. The best checkpoint is selected using validation macro-F1.

Default training settings in `train_cogswitch.py` include 10 epochs, batch size 32, learning rate `2e-5`, maximum sequence length 128, contrastive temperature 0.1, and random seed 42.

## 4. Software and external dependencies

The scripts use Python 3. The repository uses `requirements.txt` to document Python dependencies. Replace any older dependency list with the verified snapshot from the CogSwitch Linux training environment before relying on it for reproduction. Depending on the experiment, main packages include:

```
numpy
pandas
torch
transformers
datasets
scikit-learn
matplotlib
seaborn
lime
shap
anthropic
openai

```

For an environment matching the provided dependency snapshot, create a Python virtual environment and install dependencies (the commands may need adjustment for your platform):

```bash
python3 -m venv .venv
source .venv/bin/activate
python -m pip install -r requirements.txt
```

**CUDA compatibility:** The recorded environment includes `torch==2.6.0+cu124` (a CUDA 12.4 build), as well as other GPU-related packages. Some wheels require platform-specific installation sources; the full environment snapshot may not install unchanged on macOS, CPU-only systems, or different CUDA configurations. Check the [official PyTorch installation instructions](https://pytorch.org/get-started/locally/) for a compatible build. The requirements file describes the original environment; a successful clean installation on another machine has not yet been independently verified.

Dataset generation/validation scripts may additionally require FastText, a locally accessible Meta-Llama-3-8B-Instruct model, and their original source datasets. Persuasion annotation scripts require API credentials for the respective providers. Never commit credentials to the repository. Differences in software, hardware, and hosted models may affect reproducibility.

## 5. Training and evaluation

Run commands from the **repository root**. The examples below assume that `train.csv`, `val.csv`, and `test.csv` are immediately inside `dataset/`, that the documented command-line arguments are supported by the current script, and that any hard-coded paths have been updated. **These assumptions must be checked before treating the commands as fully reproducible.** If the CSVs are stored in a subdirectory, change `--data_dir` accordingly.

```bash
git clone https://github.com/JarinNooder/CogSwitch.git
cd CogSwitch
```

### Train the main model

```
python code/train_cogswitch.py --lam 0.7 --aux_weight 0.1 --data_dir dataset
```

### Repeat the lambda experiments

```
python code/train_cogswitch.py --lam 0.0 --data_dir dataset
python code/train_cogswitch.py --lam 0.3 --data_dir dataset
python code/train_cogswitch.py --lam 0.5 --data_dir dataset
python code/train_cogswitch.py --lam 0.7 --data_dir dataset
python code/train_cogswitch.py --lam 0.9 --data_dir dataset
```

Each run writes a best-checkpoint file and result history under `checkpoints/` by default. **Do not run the lambda 0.7 command twice unless you intend to overwrite its checkpoint.**

### Other experiments

| **Script**                  | **Purpose**                                                                                       | **Important input or prerequisite** |
| --------------------------- | ------------------------------------------------------------------------------------------------- | ----------------------------------- |
| `baselines.py`              | TF-IDF classical baselines and mBERT                                                              | Training and test CSVs              |
| `ablation_aux.py`           | Compare auxiliary-loss weights at lambda 0.7                                                      | Training, validation, and test CSVs |
| `fairness_eval_v2_fixed.py` | Per-language-pair error rates and equalized-odds differences, with bootstrap confidence intervals | Five saved lambda checkpoints       |
| `adversarial_robustness.py` | Homoglyph and character-perturbation tests                                                        | Lambda 0.7 checkpoint and test CSV  |
| `confusion_matrices.py`     | Language-pair confusion matrices                                                                  | Lambda 0.7 checkpoint and test CSV  |
| `shap_explain_corrected.py` | Word-level SHAP analysis on 200 sampled test messages                                             | Lambda 0.7 checkpoint and test CSV  |
| `lime_explain_revised.py`   | Illustrative LIME explanations for three language pairs                                           | Lambda 0.7 checkpoint and test CSV  |
| `tsne_viz_corrected.py`     | Qualitative encoder-embedding visualization                                                       | Lambda 0.0 and 0.7 checkpoints      |

**Path portability warning:** Some evaluation and explainability scripts contain absolute paths from the original compute environment. Those paths must be replaced with repository-relative or local paths before execution. This README does not guarantee that every script runs without modification. Review the scripts and saved output metadata for paths, credentials, and private identifiers before sharing.

**Checkpoint availability:** The dataset CSVs and saved explanation JSONs are not trained model checkpoints. Unless the required `.pt` checkpoints have been separately published, experiments depending on them must first train those models. Exact reproduction may depend on checkpoint selection, preprocessing, and the original environment.

## 6. Saved explainability outputs

### `results/shap_results.json`

Saved aggregate word-level SHAP output from `shap_explain_corrected.py`, based on **200 fixed-seed test examples**. It includes the top tokens, their mean absolute SHAP attributions **conditional on token occurrence**, and occurrence counts (minimum two). These values should not be interpreted as causal evidence or as direct measurements of persuasion mechanisms.

### `results/lime_explanations.json`

Saved local LIME explanations for illustrative code-switched phishing messages from **English–Hindi, English–Arabic, and English–Spanish**. It records the selected texts, predicted phishing probabilities, token weights, local surrogate fidelity, sampling settings, and checkpoint metadata. English–Chinese is not included in the three-panel illustration.

**Interpretation caution:** The saved LIME explanations have very low local surrogate R² values (approximately 0.004–0.019) and small token weights. They are illustrative diagnostic outputs, **not reliable evidence of strong token-level causal effects**.

The saved LIME JSON may retain an original absolute checkpoint path. Check and remove private path metadata before sharing; preserve analytical values and record any metadata-only sanitization.

## 7. Dataset construction and annotation provenance

Where included, the data-construction scripts document generation of code-mixed phishing messages using Llama 3, validation of generated messages, and assembly of the final splits. Relevant scripts include `08_generate_cs_v6_split.py`, `04e_validate_v6_splits.py`, and `10_assemble_dataset_v2_fixed.py`.

The generation process depends on source datasets and local model resources that are not necessarily bundled in this repository. Source-data licensing and redistribution conditions should be checked independently. Earlier generator versions, if retained, are provenance records and should not be mistaken for the final v6 pipeline.

The annotation scripts include Claude- and GPT-based persuasion labeling, inter-annotator agreement analysis, and an authority-label re-annotation procedure. Re-running API annotation may incur costs and can produce different outputs as hosted models change. The authority re-annotation script updates CSV files in place; work on copies rather than on the released dataset.

## 8. Reproducibility and responsible use

- The data contain phishing/social-engineering examples for defensive research. Do not use them to target real users.
- A fixed random seed does not guarantee bitwise identical results across GPU types, library versions, or external generative-model APIs.
- Language-pair labels are not demographic or nationality labels. Reported group differences should not be interpreted as individual-level fairness guarantees.
- Some generated messages may contain model artifacts, explanatory notes, or imperfect code-mixing. Inspect the examples and account for these limitations when interpreting results.
- Do not publish API keys, credentials, private URLs, personal file paths, or other sensitive metadata.
- Third-party datasets and pretrained models retain their original terms and may not be redistributable under this repository's code license.

## 9. Citation

If you use CogSwitch in research, please cite the associated paper. Machine-readable citation metadata is available in [`CITATION.cff`](CITATION.cff). Add the final publication venue, DOI, and bibliographic details when they are available.

## 10. License

The repository includes an [MIT License](LICENSE). It covers original source code only to the extent that the applicable copyright holders have authorized its release under that license. Third-party datasets, pretrained models, and other external content remain subject to their respective licenses and redistribution requirements.