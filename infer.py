# infer.py
#
# Load a pre-trained VAE-GSM model and run inference on a (possibly different)
# dataset.  No training happens — this is pure transform + evaluate.
#
# Primary use-case:
#   Train on clean data → infer on noisy data
#   to test whether noise removal during training generalises at inference time.
#
# Usage:
#   python infer.py \
#       --model_path  models/<stable_model_id>/model \
#       --dataset_cfg data_config/stackoverflow_noise.json \
#       --run_tag     clean_model_noise_infer \
#       [--seed 42]
#
# Outputs (same layout as main.py):
#   results_StackOverflow/<run_id>/
#       metrics.json          ← NMI, purity, NPMI, diversity, topic_diversity
#       topics_top_words.csv
#       doc_topic.csv
#       training_log.csv      ← skipped (no training)
#       artifacts/
#           doc_topic.npy
#           topic_word.npy
#           topic_word_prob.npy
#           topic_vectors.npy
#           word_vectors.npy
#           id2word.json
#           topics_words.json
#
# After running, point analyse_vae.py at the new run_dir:
#   python analyse_vae.py \
#       --run_dir results_StackOverflow/<run_id> \
#       --data_csv path/to/noisy_data.csv

import argparse
import json
import time
from pathlib import Path

import numpy as np

from src.utils.seed import set_seed
from src.utils.io import save_json, save_lines
from src.utils.helpers import ensure_dir, log_to_file
from src.preprocess import load_dataset
from src.evaluate_models import evaluate_and_save


# =============================================================================
# helpers (mirrors main.py without the training path)
# =============================================================================

def make_run_id(run_tag: str, dataset_name: str) -> str:
    """Timestamped run identifier so results don't overwrite each other."""
    from datetime import datetime
    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    return f"{ts}_{run_tag}_{dataset_name}"


# =============================================================================
# MAIN
# =============================================================================

def run(model_path: str, dataset_cfg_path: str, run_tag: str, seed: int = 42):
    """
    Load a saved model, transform a dataset, evaluate, save everything.

    Parameters
    ----------
    model_path      : path prefix passed to model.load() — same value that
                      was passed to model.save() in main.py, e.g.
                      "models/vae_gsm_stackoverflow_seed42_abc123/model"
    dataset_cfg_path: JSON config for the inference dataset (can be the noisy
                      version of the training dataset, or a completely different
                      dataset as long as the vocabulary overlaps)
    run_tag         : short descriptive name embedded in the results folder,
                      e.g. "clean_model_noise_infer"
    seed            : random seed for reproducibility
    """
    set_seed(seed)

    # ── 1. Load dataset config ────────────────────────────────────────────────
    cfg_ds       = json.loads(Path(dataset_cfg_path).read_text())
    dataset_name = cfg_ds["name"]

    print(f"[infer] loading dataset: {dataset_name}")
    ds   = load_dataset(cfg_ds)
    docs = ds["docs"]
    y    = ds["labels"]
    print(f"[infer] {len(docs)} documents, {len(set(y))} unique labels")

    # ── 2. Set up results directory ───────────────────────────────────────────
    run_id  = make_run_id(run_tag, dataset_name)
    res_dir = Path("results_StackOverflow") / run_id
    ensure_dir(res_dir)

    log_fp = res_dir / "logs.txt"
    t0     = time.time()
    log_to_file(log_fp,
        f"[infer] run_id={run_id}  model={model_path}  "
        f"dataset={dataset_name}  seed={seed}")

    # ── 3. Load model ─────────────────────────────────────────────────────────
    # Infer model type from the _meta.json saved alongside the weights.
    # Currently only VAE-GSM is supported; add elif blocks for LDA/NMF if needed.
    meta_path = Path(model_path + "_meta.json")
    if not meta_path.exists():
        raise FileNotFoundError(
            f"Cannot find model meta file at {meta_path}\n"
            f"Check that --model_path points to the prefix used in model.save(), "
            f"e.g. 'models/<stable_id>/model' (without any extension)."
        )

    with open(meta_path) as f:
        meta = json.load(f)

    # Detect model type — currently VAE-GSM writes a "k" key in meta
    # Add further heuristics here if you add more model types.
    if "k" in meta and "hidden_dims" in meta:
        from src.models.vae_gsm_model import VAEGSMModel
        model      = VAEGSMModel()
        model_type = "vae_gsm"
    else:
        raise ValueError(
            "Could not determine model type from meta.json. "
            "Only VAE-GSM is currently supported by infer.py."
        )

    print(f"[infer] detected model type: {model_type}")
    print(f"[infer] loading weights from: {model_path}")
    model.load(model_path)
    log_to_file(log_fp, f"[infer] model loaded (K={model.k})")

    # ── 4. Transform ──────────────────────────────────────────────────────────
    # VAE-GSM handles its own vectorization internally using the vocabulary
    # built during training. Words not in the training vocabulary are silently
    # ignored (CountVectorizer behaviour).
    print(f"[infer] transforming {len(docs)} documents ...")
    doc_topic    = model.transform(docs)      # List[List[float]], shape (N, K)
    topics_words = model.topics_top_words(topn=10)
    minutes      = (time.time() - t0) / 60.0
    print(f"[infer] transform complete in {minutes:.2f} min")

    # ── 5. Save artifacts (identical layout to main.py) ──────────────────────
    art_dir = res_dir / "artifacts"
    ensure_dir(art_dir)

    np.save(art_dir / "doc_topic.npy",
            np.asarray(doc_topic, dtype=np.float32))
    save_json(topics_words, art_dir / "topics_words.json")

    art = model.get_artifacts()
    save_json(art["id2word"],          art_dir / "id2word.json")
    np.save(art_dir / "topic_word.npy",
            np.asarray(art["topic_word"],      dtype=np.float32))
    np.save(art_dir / "topic_word_prob.npy",
            np.asarray(art["topic_word_prob"], dtype=np.float32))

    if "topic_vectors" in art and art["topic_vectors"] is not None:
        np.save(art_dir / "topic_vectors.npy",
                np.asarray(art["topic_vectors"], dtype=np.float32))

    if "word_vectors" in art and art["word_vectors"] is not None:
        np.save(art_dir / "word_vectors.npy",
                np.asarray(art["word_vectors"], dtype=np.float32))

    # ── 6. Human-readable CSVs ────────────────────────────────────────────────
    lines = ["topic,word1,word2,word3,word4,word5,word6,word7,word8,word9,word10"]
    for i, words in enumerate(topics_words):
        row  = [str(i)] + [w.replace(",", " ") for w in words[:10]]
        row += ["" for _ in range(max(0, 11 - len(row)))]
        lines.append(",".join(row))
    save_lines(lines, res_dir / "topics_top_words.csv")

    if len(doc_topic) and len(doc_topic[0]):
        header = "doc_id," + ",".join([f"t{j}" for j in range(len(doc_topic[0]))])
    else:
        header = "doc_id"
    doc_lines = [header]
    for i, dist in enumerate(doc_topic):
        doc_lines.append(str(i) + "," + ",".join(f"{float(p):.6f}" for p in dist))
    save_lines(doc_lines, res_dir / "doc_topic.csv")

    # ── 7. Evaluate ───────────────────────────────────────────────────────────
    # evaluate_and_save expects the method cfg dict; we reconstruct a minimal
    # one from the loaded meta so the evaluator gets K and other needed fields.
    cfg_m_inferred = {
        "name": model_type,
        "k":    model.k,
        "params": {
            k: meta[k] for k in [
                "k", "hidden_dims", "topic_dim", "dropout",
                "max_vocab", "normalize_input",
                "diversity_lambda", "kl_anneal_epochs",
            ] if k in meta
        }
    }

    print("[infer] evaluating ...")
    metrics = evaluate_and_save(
        model_type, dataset_name, cfg_m_inferred, res_dir,
        docs=docs, labels=y
    )
    metrics.update({
        "time_min":    round(minutes, 3),
        "run_id":      run_id,
        "seed":        seed,
        "model_path":  str(model_path),   # track which model was used
        "infer_mode":  True,              # flag: this run did not train
    })
    save_json(metrics, res_dir / "metrics.json")

    log_to_file(log_fp,
        f"[infer] finished in {minutes:.3f} min; "
        f"metrics saved to {res_dir / 'metrics.json'}")
    print(json.dumps(metrics, indent=2))
    return metrics


# =============================================================================
# CLI
# =============================================================================

if __name__ == "__main__":
    ap = argparse.ArgumentParser(
        description=(
            "Run inference with a pre-trained topic model on a (possibly different) "
            "dataset. No training is performed. Outputs mirror main.py so the results "
            "folder works directly with analyse_vae.py."
        )
    )
    ap.add_argument(
        "--model_path", required=True,
        help=(
            "Path prefix of the saved model, e.g. "
            "'models/vae_gsm_stackoverflow_seed42_abc123/model'. "
            "This is the same value passed to model.save() in main.py."
        )
    )
    ap.add_argument(
        "--dataset_cfg", required=True,
        help="Path to dataset JSON config, e.g. data_config/stackoverflow_noise.json"
    )
    ap.add_argument(
        "--run_tag", default="infer",
        help=(
            "Short label embedded in the results folder name. "
            "Use something descriptive, e.g. 'clean_model_on_noise'. "
            "Default: 'infer'"
        )
    )
    ap.add_argument(
        "--seed", type=int, default=42,
        help="Random seed. Default: 42"
    )

    args = ap.parse_args()
    run(
        model_path      = args.model_path,
        dataset_cfg_path= args.dataset_cfg,
        run_tag         = args.run_tag,
        seed            = args.seed,
    )