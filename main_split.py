####### This is streamlit version
# this is main.py
import argparse, json, time
from pathlib import Path
import numpy as np

from src.utils.seed import set_seed, make_run_id
from src.utils.io import save_json, save_lines
from src.utils.helpers import ensure_dir, log_to_file
from src.preprocess import load_dataset

from src.models.lda_model import LDAModel
from src.evaluate_models import evaluate_and_save

from sklearn.feature_extraction.text import ENGLISH_STOP_WORDS
from sklearn.model_selection import train_test_split
import re
import hashlib


def remove_stopwords_from_docs(docs):
    cleaned = []
    for doc in docs:
        tokens = re.findall(r"[a-zA-Z]+", doc.lower())
        tokens = [t for t in tokens if t not in ENGLISH_STOP_WORDS and len(t) > 2]
        cleaned.append(" ".join(tokens))
    return cleaned


def build_method(cfg, res_dir):
    name = cfg.get("name")

    if name == "lda":
        return LDAModel(
            k=cfg.get("k", 50),
            passes=cfg.get("passes", 10),
            iterations=cfg.get("iterations", 200),
            random_state=cfg.get("random_state", 42),
        )

    elif name == "vae_gsm":
        from src.models.vae_gsm_model import VAEGSMModel
        params = dict(cfg.get("params", {}))
        params["log_dir"] = str(res_dir)
        return VAEGSMModel(**params)
    elif name == "ecrtm":
        from src.models.ecrtm_model import ECRTMModel
        params = dict(cfg.get("params", {}))
        params["log_dir"] = str(res_dir)
        return ECRTMModel(**params)

    elif name == "fastopic":
        from src.models.fastopic_model import FASTopicModel
        params = dict(cfg.get("params", {}))
        return FASTopicModel(**params)

    elif name == "glocom":
        from src.models.glocom_model import GloCOMModel
        params = dict(cfg.get("params", {}))
        params["log_dir"] = str(res_dir)
        return GloCOMModel(**params)

    elif name == "ecrtm2":
        from src.models.ecrtm_2 import ECRTMTopMostModel
        return ECRTMTopMostModel(**cfg.get("params", {}))

    elif name == "pvtm":
        from src.models.pvtm_model import PVTMModel
        params = dict(cfg.get("params", {}))
        params["log_dir"] = str(res_dir)
        return PVTMModel(**params)

    else:
        raise ValueError(f"Unknown method: {name}")


def make_stable_model_id(method_name: str, dataset_name: str, cfg_m: dict, seed: int) -> str:
    flat_excluded = {"name"}

    if "params" in cfg_m:
        params = cfg_m["params"]
    else:
        params = {k: v for k, v in cfg_m.items() if k not in flat_excluded}

    params_str = json.dumps(params, sort_keys=True, separators=(",", ":"))
    h = hashlib.sha1(params_str.encode("utf-8")).hexdigest()[:10]
    return f"{method_name}_{dataset_name}_seed{seed}_{h}"


def run(dataset_cfg_path: str, method_cfg_path: str, seed: int = 42,
        test_ratio: float = 0.2):
    cfg_ds = json.loads(Path(dataset_cfg_path).read_text())
    cfg_m  = json.loads(Path(method_cfg_path).read_text())
    set_seed(seed)

    # Load dataset
    ds = load_dataset(cfg_ds)
    docs_all, y_all = ds["docs"], ds["labels"]
    eval_docs_all = ds.get("eval_docs") or docs_all
    dataset_name = cfg_ds["name"]
    method_name  = cfg_m["name"]

    # ------------------------------------------------------------------
    # Stratified train/test split (80/20 per class by default)
    # ------------------------------------------------------------------
    n_all = len(docs_all)
    all_idx = np.arange(n_all)

    if y_all is not None and test_ratio > 0:
        train_idx, test_idx = train_test_split(
            all_idx, test_size=test_ratio,
            stratify=y_all, random_state=seed,
        )
    elif test_ratio > 0:
        train_idx, test_idx = train_test_split(
            all_idx, test_size=test_ratio,
            random_state=seed,
        )
    else:
        train_idx = all_idx
        test_idx = np.array([], dtype=np.int64)

    # Sort indices so doc order within each split is deterministic
    train_idx = np.sort(train_idx)
    test_idx  = np.sort(test_idx)

    # Slice docs, eval_docs, labels for each split
    docs_train      = [docs_all[i] for i in train_idx]
    docs_eval_train = [eval_docs_all[i] for i in train_idx]
    y_train         = [y_all[i] for i in train_idx] if y_all is not None else None

    docs_eval_test  = [eval_docs_all[i] for i in test_idx] if len(test_idx) > 0 else []
    y_test          = [y_all[i] for i in test_idx] if (y_all is not None and len(test_idx) > 0) else None

    # RESULTS: keep per-run (timestamped)
    run_id  = make_run_id(method_name, dataset_name)
    res_dir = Path("results_Marginal") / run_id
    ensure_dir(res_dir)

    # LOG file
    log_fp = res_dir / "logs.txt"
    t0 = time.time()
    log_to_file(log_fp, f"[run] start run_id={run_id} method={method_name} "
                         f"dataset={dataset_name} seed={seed} "
                         f"n_train={len(train_idx)} n_test={len(test_idx)}")

    # MODELS: stable across runs
    stable_model_id = make_stable_model_id(method_name, dataset_name, cfg_m, seed)
    mod_dir    = Path("models") / stable_model_id
    ensure_dir(mod_dir)
    model_path = mod_dir / "model"

    # ------------------------------------------------------------------
    # Method-specific preprocessing (only on training docs)
    # ------------------------------------------------------------------
    if method_name in {"lda", "bertopic"}:
        log_to_file(log_fp, "[preprocess] removing stopwords")
        docs_train = remove_stopwords_from_docs(docs_train)

    # Build model
    model = build_method(cfg_m, res_dir)

    can_resume = method_name in {
        "lda", "nmf", "senclu", "senclu_fttopic",
        "bertopic", "sentbpe", "vae_gsm", "ecrtm", "pvtm"
    }

    if can_resume and model_path.exists():
        log_to_file(log_fp, f"[model] loading existing model from {model_path}")
        model.load(model_path)
    else:
        log_to_file(log_fp, "[model] training on train split only")
        model.fit(docs_train)
        model.save(model_path)

    # ------------------------------------------------------------------
    # Transform BOTH splits (using eval text, not train text)
    # ------------------------------------------------------------------
    doc_topic_train = model.transform(docs_eval_train)
    topics_words    = model.topics_top_words(topn=10)

    doc_topic_test = None
    if len(test_idx) > 0:
        doc_topic_test = model.transform(docs_eval_test)

    minutes = (time.time() - t0) / 60.0

    # ------------------------------------------------------------------
    # Save analysis-ready artifacts (model-agnostic)
    # ------------------------------------------------------------------
    art_dir = res_dir / "artifacts"
    ensure_dir(art_dir)

    # Train theta (backward compatible: doc_topic.npy = train)
    np.save(art_dir / "doc_topic.npy",
            np.asarray(doc_topic_train, dtype=np.float32))
    save_json(topics_words, art_dir / "topics_words.json")

    # Test theta + split indices
    if doc_topic_test is not None:
        np.save(art_dir / "doc_topic_test.npy",
                np.asarray(doc_topic_test, dtype=np.float32))
    np.save(art_dir / "train_indices.npy", train_idx.astype(np.int64))
    np.save(art_dir / "test_indices.npy",  test_idx.astype(np.int64))

    # Split metadata for the UI
    split_info = {
        "test_ratio":  test_ratio,
        "seed":        seed,
        "n_total":     int(n_all),
        "n_train":     int(len(train_idx)),
        "n_test":      int(len(test_idx)),
    }
    save_json(split_info, art_dir / "split_info.json")

    art = model.get_artifacts()

    save_json(art["id2word"], art_dir / "id2word.json")
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

    if "doc_embeddings" in art and art["doc_embeddings"] is not None:
        np.save(art_dir / "doc_embeddings.npy",
                np.asarray(art["doc_embeddings"], dtype=np.float32))

    # ------------------------------------------------------------------
    # Human-readable CSVs
    # ------------------------------------------------------------------
    lines = ["topic,word1,word2,word3,word4,word5,word6,word7,word8,word9,word10"]
    for i, words in enumerate(topics_words):
        row  = [str(i)] + [w.replace(",", " ") for w in words[:10]]
        row += ["" for _ in range(max(0, 11 - len(row)))]
        lines.append(",".join(row))
    save_lines(lines, res_dir / "topics_top_words.csv")

    if len(doc_topic_train) and len(doc_topic_train[0]):
        header = "doc_id," + ",".join([f"t{j}" for j in range(len(doc_topic_train[0]))])
    else:
        header = "doc_id"
    doc_lines = [header]
    for i, dist in enumerate(doc_topic_train):
        if isinstance(dist, (list, tuple)):
            doc_lines.append(str(i) + "," + ",".join(f"{float(p):.6f}" for p in dist))
        else:
            doc_lines.append(str(i))
    save_lines(doc_lines, res_dir / "doc_topic.csv")

    # ------------------------------------------------------------------
    # Evaluate on TRAIN split (backward compatible)
    # ------------------------------------------------------------------
    metrics = evaluate_and_save(
        method_name, dataset_name, cfg_m, res_dir,
        docs=docs_eval_train, labels=y_train
    )

    # ------------------------------------------------------------------
    # Evaluate clustering on TEST split too (if available)
    # ------------------------------------------------------------------
    if doc_topic_test is not None and y_test is not None:
        from src.evaluate_models import compute_nmi, evaluate_clustering
        test_nmi = compute_nmi(y_test, np.array(doc_topic_test))
        test_clust = evaluate_clustering(np.asarray(doc_topic_test), np.asarray(y_test))
        metrics["test_nmi"]    = round(float(test_nmi), 4) if test_nmi is not None else None
        metrics["test_purity"] = round(float(test_clust["purity"]), 4)

    metrics.update({
        "time_min":        round(minutes, 3),
        "run_id":          run_id,
        "seed":            seed,
        "stable_model_id": stable_model_id,
        "n_train":         int(len(train_idx)),
        "n_test":          int(len(test_idx)),
        "test_ratio":      test_ratio,
    })
    save_json(metrics, res_dir / "metrics.json")

    # ------------------------------------------------------------------
    # Dataset fingerprint
    # ------------------------------------------------------------------
    fingerprint = {
        "dataset_name":  dataset_name,
        "file_name":     cfg_ds.get("file_name"),
        "min_doc_len":   int(cfg_ds.get("min_doc_len", 0)),
        "labels_flag":   bool(cfg_ds.get("labels", False)),
        "n_docs":        int(len(train_idx)),   # doc_topic.npy rows = train
        "n_docs_total":  int(n_all),
        "n_test":        int(len(test_idx)),
        "test_ratio":    test_ratio,
        "vocab_size":    int(len(art["id2word"])),
        "k":             int(len(topics_words)),
        "method":        method_name,
        "seed":          int(seed),
        "used_extended_text":     (ds.get("eval_docs") is not None),
        "train_eval_text_differ": (ds.get("eval_docs") is not None),
    }
    save_json(fingerprint, res_dir / "dataset_fingerprint.json")

    log_to_file(log_fp,
        f"[run] finished in {minutes:.3f} min; "
        f"metrics saved to {res_dir / 'metrics.json'}")
    print(json.dumps(metrics, indent=2))
    return metrics


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset_cfg", required=True)
    ap.add_argument("--method_cfg",  required=True)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--test_ratio", type=float, default=0,
                    help="Fraction of docs held out per class (0 = no split)")
    args = ap.parse_args()
    run(args.dataset_cfg, args.method_cfg, args.seed, args.test_ratio)