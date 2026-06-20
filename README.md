# Neural Topic Modeling

### LLM-Based Selective Extension and Mixture-of-Experts Routing

Neural topic models (NTMs) built on variational autoencoders struggle on **short, multi-label, class-imbalanced** corpora. A single encoder bottleneck routes capacity to the majority classes, so three failures recur:

| Failure | Symptom |
|---|---|
| **Topic absence** | A rare class gets no dedicated topic; its vocabulary is absorbed by a larger neighbour. |
| **Topic collision** | Two distinct fields fuse into one topic; their document–topic vectors become inseparable. |
| **No interpretability** | The encoder is a black-box MLP over a bag-of-words; no per-word attribution explains a topic assignment. |

This repository attacks all three from three complementary angles ("threads"). The full write-up is in [`docs/report.pdf`](docs/report.pdf).

| Thread | Angle | Idea | Code |
|---|---|---|---|
| **1 — LLM Document Extension** | Data | Use an instruction-tuned LLM to extend documents, either **only for rare classes (`+sel`)** or **for every class (`+all`)**, supplying the missing co-occurrence signal. | `augmentation/`, `scripts/llm_extension/`, `src/models/vae_gsm_use_model.py` |
| **2 — MoE-NTM** | Architecture | Replace the single VAE encoder with a **Mixture-of-Experts** encoder; compare dense / sparse top-*k* / expert-choice document-level gating. | `src/models/moe_ntm*.py` |
| **3 — PWAE-NTM** | Deeper architecture | Push routing to the **word level**: each expert attends over a document's words with a learnable query, making routing traceable to the deciding tokens. | `src/models/pwae_ntm_model.py`, `scripts/analysis/analyse_pwae_*.py` |

---

## Repository layout

```
NEURAL TOPIC MODELING/
├── main.py                 # orchestrator: load data → build model → train → save → evaluate
├── main_split.py           # train/test-split variant (saves doc_topic_test.npy)
├── infer.py                # encode a new corpus with an already-trained model
│
├── src/
│   ├── preprocess.py       # dataset loading + BoW / embedding vectorisation
│   ├── evaluate_models.py  # NPMI / C_V / NMI / Purity / RF-macro-F1 + artifact dump
│   ├── models/
│   │   ├── base.py                     # BaseTM interface every model implements
│   │   ├── vae_gsm_model.py            # VAE-GSM baseline (Miao et al. 2017)
│   │   ├── vae_gsm_use_model.py        # SBERT/USE-input VAE-GSM + hinge-gated β-diversity (Thread 1)
│   │   ├── moe_ntm_model.py            # dense MoE encoder           ┐
│   │   ├── moe_ntm_sparse_model.py     # sparse top-k gating         │ Thread 2
│   │   ├── moe_ntm_ec_model.py         # expert-choice routing       │ (each has a *_use_* SBERT variant)
│   │   ├── moe_ntm_attn_model.py       # attention gating            ┘
│   │   ├── pwae_ntm_model.py           # PWAE-NTM word-level attention experts (Thread 3)
│   │   ├── wlr_*_ntm_model.py          # word-level-routed NTM variants (exploratory)
│   │   └── {lda,ecrtm,ecrtm_2,fastopic,glocom,pvtm}_model.py   # baselines
│   └── utils/              # seeds, IO, embeddings, tokenisation helpers
│
├── data_config/            # one JSON per dataset (path, label columns, vocab size)
├── model_config/           # one JSON per model/hyper-parameter variant
│
├── augmentation/           # Thread 1 library: rare-cluster detection + LLM expansion pipeline
│   ├── cluster_detector.py # find rare semantic clusters (HDBSCAN over SBERT)
│   ├── expander.py         # per-document LLM rewrite
│   └── pipeline.py / evaluate.py
│
├── scripts/
│   ├── llm_extension/      # generate_extended_text.py, jsnol_csv.py, merge_extended_tweet.py
│   ├── experiments/        # run_all_experiments.py, run_pwae_ntm.py, report generators
│   └── analysis/           # PWAE attention / domain-specialist / routing / classification analyses
│
├── analysis_ui/            # read-only Streamlit dashboard (Overview, Doc Inspector, …)
├── docs/report.pdf         # the thesis this repo accompanies
├── requirements.txt
└── .gitignore              # data/, models/, results_*/ are regenerated, not versioned
```

> **Note.** Large artifacts — `data/`, `models/`, `cache/`, `results_*/` — are intentionally **not** included (see `.gitignore`). The code regenerates them. Place your datasets under `data/` and point the `data_config/*.json` files at them.

---

## Installation

```bash
python -m venv .venv && source .venv/bin/activate     # or conda create -n ntm python=3.10
pip install -r requirements.txt
```

Tested with Python 3.10 and PyTorch ≥ 2.0 (CUDA optional but recommended).

---

## Usage

Every model follows the same `BaseTM` interface and is selected by a `model_config/*.json` + `data_config/*.json` pair.

### Train a model

```bash
# VAE-GSM baseline on GoogleNews
python main.py --dataset_cfg data_config/googlenewst.json \
               --method_cfg  model_config/vae_gsm.json --seed 42

# MoE-NTM, expert-choice routing on Reuters-10
CUDA_VISIBLE_DEVICES=0 python main.py \
               --dataset_cfg data_config/reuters_10.json \
               --method_cfg  model_config/moe_ntm_ec_k10.json \
               --results_dir results_reuters_10 --seed 42

# PWAE-NTM (best config) on Reuters-10
CUDA_VISIBLE_DEVICES=0 python main.py \
               --dataset_cfg data_config/reuters_10.json \
               --method_cfg  model_config/pwae_ntm_reuters10_v4c_full_tfidf_stop.json \
               --results_dir results_reuters_10_pwae --seed 42
```

Each run writes a timestamped folder under `results_<Dataset>/` containing `metrics.json`, `topics_top_words.csv`, `doc_topic.csv`, and an `artifacts/` directory (`doc_topic.npy`, `topic_word_prob.npy`, `gate_weights.npy`, …). Unchanged configs reload a cached model instead of retraining.

### Thread 1 — LLM document extension

```bash
# 1. Generate extended text with Mistral-7B-Instruct
python scripts/llm_extension/generate_extended_text.py \
       --dataset_cfg data_config/googlenewst.json \
       --model_dir   /path/to/Mistral-7B-Instruct \
       --output_dir  LLM/extended --batch_size 16 --gpus 0 1

# 2. Convert JSONL → CSV and merge only the rare clusters (+sel)
python scripts/llm_extension/jsnol_csv.py --input LLM/extended/googlenewst_mistral.jsonl \
       --output data/googlenewst_filled.csv --minimal
python scripts/llm_extension/merge_extended_tweet.py     # RARE_CLUSTERS controls +sel vs +all

# 3. Train VAE-GSM on the extended corpus (see model_config/vae_gsm_use_*.json)
```

### Evaluate / analyse a run

```bash
python scripts/analysis/classify_multi.py \
       --theta results_reuters_10_pwae/<run>/artifacts/doc_topic.npy \
       --csv   data/reuters_10.csv --label_start_col interest

python scripts/analysis/analyse_pwae_correct_docs.py    # word-level routing walkthroughs
python scripts/analysis/analyse_domain_specialist.py    # domain-relevance lift (gate > 0.6)

streamlit run analysis_ui/app.py                        # interactive dashboard
```

---

## Datasets

| Dataset | Docs | Avg. words | Classes | Vocab | Used in |
|---|---|---|---|---|---|
| GoogleNews-10 | 2,414 | 6.3 | 10 | 500 | Thread 1 |
| Tweet-10 | 1,205 | 8.7 | 10 | 500 | Thread 1 |
| Reuters-10 | 2,929 | 187.8 | 10 (multi-label) | 2000 | Threads 2, 3 |
| 20news-5class | 4,778 | 188.0 | 5 | 2000 | Thread 3 |

## Evaluation metrics

- **RF macro-F1** — macro-averaged F1 of a Random Forest trained on the document–topic vectors θ. Measures class separability without assuming any topic-to-label mapping.
- **NPMI** / **C_V** — topic-word coherence against a Wikipedia reference corpus.
- **NMI**, **Purity**, **topic-diversity** — clustering quality and topic distinctness.

---

## Headline results (seed 42)

**Thread 1 — targeted extension wins on short text.** On Tweet-10, `VAE-GSM-SBERT+sel` reaches **RF 0.994** — best of an 11-model comparison. On GoogleNews-10, `+sel` closes 97 % of the baseline-to-ceiling gap (0.776 → 0.989). Blanket `+all` posts higher NPMI but injects LLM-style vocabulary and is never the best classifier — **coherence and classification are decoupled.**

**Thread 2 — expert-choice rescues the tail.** On Reuters-10, expert-choice routing gives the best macro-F1 (**0.724**) and micro-F1 (0.802); the gain comes almost entirely from the rare `nat-gas` label (0.250 → **0.546**), which only EC's guaranteed per-expert capacity recovers.

**Thread 3 — word-level routing is interpretable *and* faithful.** PWAE-NTM gives the best Reuters-10 coherence (NPMI **0.295**, C_V **0.468**) and traces every assignment to the deciding tokens. Confidently-routed documents attend to the topic-defining vocabulary **2.8–3.1×** above chance, despite the gate and decoder being untied parameters.

See `docs/report.pdf` for the full ablations, taxonomy, and per-label analysis.

---

## Key references

- Miao, Grefenstette, Blunsom. *Discovering Discrete Latent Topics with Neural Variational Inference.* ICML 2017. (VAE-GSM)
- Zhou et al. *Mixture-of-Experts with Expert Choice Routing.* NeurIPS 2022.
- Wu et al. *ECRTM* (ICML 2023), *FASTopic* (NeurIPS 2024); Nguyen et al. *GloCOM* (NAACL 2025); Akash & Chang. *PVTM* (Findings of EMNLP 2024).

---

*MTech thesis project, Department of Computer Science and Automation, Indian Institute of Science.*
