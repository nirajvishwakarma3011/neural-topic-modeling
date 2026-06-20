# -*- coding: utf-8 -*-
# # this is main.py
# import argparse, json, time
# from pathlib import Path
# import numpy as np

# from src.utils.seed import set_seed, make_run_id
# from src.utils.io import save_json, save_lines
# from src.utils.helpers import ensure_dir, log_to_file
# from src.preprocess import load_dataset

# from src.models.lda_model import LDAModel
# from src.evaluate_models import evaluate_and_save

# from sklearn.feature_extraction.text import ENGLISH_STOP_WORDS
# import re
# import hashlib


# def remove_stopwords_from_docs(docs):
#     cleaned = []
#     for doc in docs:
#         tokens = re.findall(r"[a-zA-Z]+", doc.lower())
#         tokens = [t for t in tokens if t not in ENGLISH_STOP_WORDS and len(t) > 2]
#         cleaned.append(" ".join(tokens))
#     return cleaned


# def build_method(cfg, res_dir):
#     name = cfg.get("name")

#     if name == "lda":
#         return LDAModel(
#             k=cfg.get("k", 50),
#             passes=cfg.get("passes", 10),
#             iterations=cfg.get("iterations", 200),
#             random_state=cfg.get("random_state", 42),
#         )

#     elif name == "vae_gsm":
#         from src.models.vae_gsm_model import VAEGSMModel
#         # CHANGE 2: copy params before mutating to avoid modifying the
#         # original cfg_m dict (params is a reference, not a copy)
#         params = dict(cfg.get("params", {}))
#         params["log_dir"] = str(res_dir)   # pass res_dir so training_log.csv
#                                            # lands next to metrics.json
#         return VAEGSMModel(**params)
#     elif name == "ecrtm":
#         from src.models.ecrtm_model import ECRTMModel
#         params = dict(cfg.get("params", {}))
#         params["log_dir"] = str(res_dir)   # training_log.csv lands next to metrics.json
#         return ECRTMModel(**params)

#     elif name == "fastopic":
#         from src.models.fastopic_model import FASTopicModel
#         params = dict(cfg.get("params", {}))
#         # No log_dir — FASTopic does not support it
#         return FASTopicModel(**params)

#     elif name == "glocom":
#         from src.models.glocom_model import GloCOMModel
#         params = dict(cfg.get("params", {}))
#         params["log_dir"] = str(res_dir)
#         return GloCOMModel(**params)   

#     elif name == "ecrtm2":
#         from src.models.ecrtm_2 import ECRTMTopMostModel
#         return ECRTMTopMostModel(**cfg.get("params", {}))    

#     elif name == "pvtm":
#         from src.models.pvtm_model import PVTMModel
#         params = dict(cfg.get("params", {}))
#         params["log_dir"] = str(res_dir)
#         return PVTMModel(**params)        

#     else:
#         raise ValueError(f"Unknown method: {name}")


# def make_stable_model_id(method_name: str, dataset_name: str, cfg_m: dict, seed: int) -> str:
#     flat_excluded = {"name"}

#     if "params" in cfg_m:
#         params = cfg_m["params"]
#     else:
#         params = {k: v for k, v in cfg_m.items() if k not in flat_excluded}

#     params_str = json.dumps(params, sort_keys=True, separators=(",", ":"))
#     h = hashlib.sha1(params_str.encode("utf-8")).hexdigest()[:10]
#     return f"{method_name}_{dataset_name}_seed{seed}_{h}"


# def run(dataset_cfg_path: str, method_cfg_path: str, seed: int = 42):
#     cfg_ds = json.loads(Path(dataset_cfg_path).read_text())
#     cfg_m  = json.loads(Path(method_cfg_path).read_text())
#     set_seed(seed)

#     # Load dataset
#     ds = load_dataset(cfg_ds)
#     docs, y = ds["docs"], ds["labels"]
#     dataset_name = cfg_ds["name"]
#     method_name  = cfg_m["name"]

#     # RESULTS: keep per-run (timestamped)
#     run_id  = make_run_id(method_name, dataset_name)
#     res_dir = Path("results_LLM") / run_id
#     ensure_dir(res_dir)

#     # LOG file
#     log_fp = res_dir / "logs.txt"
#     t0 = time.time()
#     log_to_file(log_fp, f"[run] start run_id={run_id} method={method_name} "
#                          f"dataset={dataset_name} seed={seed}")

#     # MODELS: stable across runs
#     stable_model_id = make_stable_model_id(method_name, dataset_name, cfg_m, seed)
#     mod_dir    = Path("models") / stable_model_id
#     ensure_dir(mod_dir)
#     model_path = mod_dir / "model"

#     # ------------------------------------------------------------------
#     # Method-specific preprocessing
#     # CHANGE 1: removed vae_gsm — CountVectorizer inside VAEGSMModel
#     # handles stopwords, so applying remove_stopwords_from_docs here
#     # would double-process the text and potentially hurt performance.
#     # ------------------------------------------------------------------
#     if method_name in {"lda", "bertopic"}:
#         log_to_file(log_fp, "[preprocess] removing stopwords")
#         docs = remove_stopwords_from_docs(docs)

#     # Build model — passes res_dir for log_dir (used by vae_gsm)
#     model = build_method(cfg_m, res_dir)

#     # CHANGE 4: added vae_gsm to can_resume so trained models are
#     # reused across runs with identical configs (same stable_model_id).
#     # VAEGSMModel has save() and load() implemented so this is safe.
#     can_resume = method_name in {
#         "lda", "nmf", "senclu", "senclu_fttopic",
#         "bertopic", "sentbpe", "vae_gsm"  , "ecrtm" , "pvtm"        
#     }

#     if can_resume and model_path.exists():
#         log_to_file(log_fp, f"[model] loading existing model from {model_path}")
#         model.load(model_path)
#     else:
#         log_to_file(log_fp, "[model] training (no saved model or not resumable)")
#         model.fit(docs)
#         model.save(model_path)

#     # Transform + topics
#     doc_topic    = model.transform(docs)
#     topics_words = model.topics_top_words(topn=10)
#     minutes      = (time.time() - t0) / 60.0

#     # ------------------------------------------------------------------
#     # Save analysis-ready artifacts (model-agnostic)
#     # ------------------------------------------------------------------
#     art_dir = res_dir / "artifacts"
#     ensure_dir(art_dir)

#     np.save(art_dir / "doc_topic.npy",
#             np.asarray(doc_topic, dtype=np.float32))
#     save_json(topics_words, art_dir / "topics_words.json")

#     art = model.get_artifacts()

#     save_json(art["id2word"], art_dir / "id2word.json")
#     np.save(art_dir / "topic_word.npy",
#             np.asarray(art["topic_word"],      dtype=np.float32))
#     np.save(art_dir / "topic_word_prob.npy",
#             np.asarray(art["topic_word_prob"], dtype=np.float32))

#     if "topic_vectors" in art and art["topic_vectors"] is not None:
#         np.save(art_dir / "topic_vectors.npy",
#                 np.asarray(art["topic_vectors"], dtype=np.float32))

#     # CHANGE 3: save word_vectors — only VAE models return this.
#     # analyse_vae.py uses it for nearest-neighbor word queries and
#     # word embedding plots. LDA/NMF will simply not have this file.
#     if "word_vectors" in art and art["word_vectors"] is not None:
#         np.save(art_dir / "word_vectors.npy",
#                 np.asarray(art["word_vectors"], dtype=np.float32))

#     if "doc_embeddings" in art and art["doc_embeddings"] is not None:
#         np.save(art_dir / "doc_embeddings.npy",
#                 np.asarray(art["doc_embeddings"], dtype=np.float32))

#     # ------------------------------------------------------------------
#     # Human-readable CSVs
#     # ------------------------------------------------------------------
#     lines = ["topic,word1,word2,word3,word4,word5,word6,word7,word8,word9,word10"]
#     for i, words in enumerate(topics_words):
#         row  = [str(i)] + [w.replace(",", " ") for w in words[:10]]
#         row += ["" for _ in range(max(0, 11 - len(row)))]
#         lines.append(",".join(row))
#     save_lines(lines, res_dir / "topics_top_words.csv")

#     if len(doc_topic) and len(doc_topic[0]):
#         header = "doc_id," + ",".join([f"t{j}" for j in range(len(doc_topic[0]))])
#     else:
#         header = "doc_id"
#     doc_lines = [header]
#     for i, dist in enumerate(doc_topic):
#         if isinstance(dist, (list, tuple)):
#             doc_lines.append(str(i) + "," + ",".join(f"{float(p):.6f}" for p in dist))
#         else:
#             doc_lines.append(str(i))
#     save_lines(doc_lines, res_dir / "doc_topic.csv")

#     # ------------------------------------------------------------------
#     # Evaluate + save metrics
#     # ------------------------------------------------------------------
#     metrics = evaluate_and_save(
#         method_name, dataset_name, cfg_m, res_dir,
#         docs=docs, labels=y
#     )
#     metrics.update({
#         "time_min":        round(minutes, 3),
#         "run_id":          run_id,
#         "seed":            seed,
#         "stable_model_id": stable_model_id,
#     })
#     save_json(metrics, res_dir / "metrics.json")

#     log_to_file(log_fp,
#         f"[run] finished in {minutes:.3f} min; "
#         f"metrics saved to {res_dir / 'metrics.json'}")
#     print(json.dumps(metrics, indent=2))
#     return metrics


# if __name__ == "__main__":
#     ap = argparse.ArgumentParser()
#     ap.add_argument("--dataset_cfg", required=True)
#     ap.add_argument("--method_cfg",  required=True)
#     ap.add_argument("--seed", type=int, default=42)
#     args = ap.parse_args()
#     run(args.dataset_cfg, args.method_cfg, args.seed)



####### This is stramlit version 
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
        # CHANGE 2: copy params before mutating to avoid modifying the
        # original cfg_m dict (params is a reference, not a copy)
        params = dict(cfg.get("params", {}))
        params["log_dir"] = str(res_dir)   # pass res_dir so training_log.csv
                                           # lands next to metrics.json
        return VAEGSMModel(**params)
    elif name == "ecrtm":
        from src.models.ecrtm_model import ECRTMModel
        params = dict(cfg.get("params", {}))
        params["log_dir"] = str(res_dir)   # training_log.csv lands next to metrics.json
        return ECRTMModel(**params)

    elif name == "fastopic":
        from src.models.fastopic_model import FASTopicModel
        params = dict(cfg.get("params", {}))
        # No log_dir — FASTopic does not support it
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

    elif name == "vae_gsm_use":
        from src.models.vae_gsm_use_model import VAEGSMUSEModel
        params = dict(cfg.get("params", {}))
        params["log_dir"] = str(res_dir)
        return VAEGSMUSEModel(**params)

    elif name == "moe_ntm":
        from src.models.moe_ntm_model import MoENTMModel
        params = dict(cfg.get("params", {}))
        params["log_dir"] = str(res_dir)
        return MoENTMModel(**params)

    elif name == "moe_ntm_use":
        from src.models.moe_ntm_use_model import MoENTMUSEModel
        params = dict(cfg.get("params", {}))
        params["log_dir"] = str(res_dir)
        return MoENTMUSEModel(**params)

    elif name == "moe_ntm_sparse":
        from src.models.moe_ntm_sparse_model import MoESparseNTMModel as MoeSparseNTMModel
        params = dict(cfg.get("params", {}))
        params["log_dir"] = str(res_dir)
        return MoeSparseNTMModel(**params)

    elif name == "moe_ntm_use_sparse":
        from src.models.moe_ntm_use_sparse_model import MoeSparseNTMUSEModel
        params = dict(cfg.get("params", {}))
        params["log_dir"] = str(res_dir)
        return MoeSparseNTMUSEModel(**params)

    elif name == "moe_ntm_attn":
        from src.models.moe_ntm_attn_model import MoEAttnNTMModel
        params = dict(cfg.get("params", {}))
        params["log_dir"] = str(res_dir)
        return MoEAttnNTMModel(**params)

    elif name == "moe_ntm_use_attn":
        from src.models.moe_ntm_use_attn_model import MoEAttnNTMUSEModel
        params = dict(cfg.get("params", {}))
        params["log_dir"] = str(res_dir)
        return MoEAttnNTMUSEModel(**params)

    elif name == "moe_ntm_ec":
        from src.models.moe_ntm_ec_model import MoEECNTMModel
        params = dict(cfg.get("params", {}))
        params["log_dir"] = str(res_dir)
        return MoEECNTMModel(**params)

    elif name == "moe_ntm_use_ec":
        from src.models.moe_ntm_use_ec_model import MoEECNTMUSEModel
        params = dict(cfg.get("params", {}))
        params["log_dir"] = str(res_dir)
        return MoEECNTMUSEModel(**params)

    elif name == "pwae_ntm":
        from src.models.pwae_ntm_model import PWAENTMModel
        params = dict(cfg.get("params", {}))
        params["log_dir"] = str(res_dir)
        return PWAENTMModel(**params)

    elif name == "wlr_clean_ntm":
        from src.models.wlr_clean_ntm_model import WLRCleanNTMModel
        params = dict(cfg.get("params", {}))
        params["log_dir"] = str(res_dir)
        return WLRCleanNTMModel(**params)

    elif name == "wlr_vae_ntm":
        from src.models.wlr_vae_ntm_model import WLRVAENTMModel
        params = dict(cfg.get("params", {}))
        params["log_dir"] = str(res_dir)
        return WLRVAENTMModel(**params)

    elif name == "wlr_ctxgate_ntm":
        from src.models.wlr_ctxgate_ntm_model import WLRCtxGateNTMModel
        params = dict(cfg.get("params", {}))
        params["log_dir"] = str(res_dir)
        return WLRCtxGateNTMModel(**params)

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
        results_dir: str = "results_Abstract"):
    cfg_ds = json.loads(Path(dataset_cfg_path).read_text())
    cfg_m  = json.loads(Path(method_cfg_path).read_text())
    set_seed(seed)

    # Load dataset
    ds = load_dataset(cfg_ds)
    docs, y = ds["docs"], ds["labels"]
    docs_eval = ds.get("eval_docs") or docs
    dataset_name = cfg_ds["name"]
    method_name  = cfg_m["name"]

    # RESULTS: keep per-run (timestamped)
    run_id  = make_run_id(method_name, dataset_name)
    res_dir = Path(results_dir) / run_id
    ensure_dir(res_dir)

    # LOG file
    log_fp = res_dir / "logs.txt"
    t0 = time.time()
    log_to_file(log_fp, f"[run] start run_id={run_id} method={method_name} "
                         f"dataset={dataset_name} seed={seed}")

    # Inject CLI seed into params["random_state"] so model training uses
    # the same seed as the run, not the hardcoded config value.
    if "params" in cfg_m and "random_state" in cfg_m["params"]:
        cfg_m = dict(cfg_m)
        cfg_m["params"] = dict(cfg_m["params"])
        cfg_m["params"]["random_state"] = seed

    # MODELS: stable across runs
    stable_model_id = make_stable_model_id(method_name, dataset_name, cfg_m, seed)
    mod_dir    = Path("models") / stable_model_id
    ensure_dir(mod_dir)
    model_path = mod_dir / "model"

    # ------------------------------------------------------------------
    # Method-specific preprocessing
    # CHANGE 1: removed vae_gsm — CountVectorizer inside VAEGSMModel
    # handles stopwords, so applying remove_stopwords_from_docs here
    # would double-process the text and potentially hurt performance.
    # ------------------------------------------------------------------
    if method_name in {"lda", "bertopic"}:
        log_to_file(log_fp, "[preprocess] removing stopwords")
        docs = remove_stopwords_from_docs(docs)

    # Build model — passes res_dir for log_dir (used by vae_gsm)
    model = build_method(cfg_m, res_dir)

    # CHANGE 4: added vae_gsm to can_resume so trained models are
    # reused across runs with identical configs (same stable_model_id).
    # VAEGSMModel has save() and load() implemented so this is safe.
    can_resume = method_name in {
        "lda", "nmf", "senclu", "senclu_fttopic",
        "bertopic", "sentbpe", "vae_gsm", "ecrtm", "ecrtm2", "pvtm", "vae_gsm_use", "moe_ntm", "moe_ntm_use",
        "moe_ntm_sparse", "moe_ntm_use_sparse",
        "moe_ntm_attn", "moe_ntm_use_attn",
        "moe_ntm_ec", "moe_ntm_use_ec",
        "pwae_ntm", "wlr_clean_ntm", "wlr_vae_ntm", "wlr_ctxgate_ntm",
    }

    if can_resume and model_path.exists():
        log_to_file(log_fp, f"[model] loading existing model from {model_path}")
        model.load(model_path)
    else:
        log_to_file(log_fp, "[model] training (no saved model or not resumable)")
        model.fit(docs)
        model.save(model_path)

    # Transform + topics
    # doc_topic    = model.transform(docs)
    doc_topic    = model.transform(docs_eval)
    topics_words = model.topics_top_words(topn=10)
    minutes      = (time.time() - t0) / 60.0

    # ------------------------------------------------------------------
    # Save analysis-ready artifacts (model-agnostic)
    # ------------------------------------------------------------------
    art_dir = res_dir / "artifacts"
    ensure_dir(art_dir)

    np.save(art_dir / "doc_topic.npy",
            np.asarray(doc_topic, dtype=np.float32))
    save_json(topics_words, art_dir / "topics_words.json")

    art = model.get_artifacts()

    save_json(art["id2word"], art_dir / "id2word.json")
    np.save(art_dir / "topic_word.npy",
            np.asarray(art["topic_word"],      dtype=np.float32))
    np.save(art_dir / "topic_word_prob.npy",
            np.asarray(art["topic_word_prob"], dtype=np.float32))

    if "topic_vectors" in art and art["topic_vectors"] is not None:
        np.save(art_dir / "topic_vectors.npy",
                np.asarray(art["topic_vectors"], dtype=np.float32))

    # CHANGE 3: save word_vectors — only VAE models return this.
    # analyse_vae.py uses it for nearest-neighbor word queries and
    # word embedding plots. LDA/NMF will simply not have this file.
    if "word_vectors" in art and art["word_vectors"] is not None:
        np.save(art_dir / "word_vectors.npy",
                np.asarray(art["word_vectors"], dtype=np.float32))

    if "doc_embeddings" in art and art["doc_embeddings"] is not None:
        np.save(art_dir / "doc_embeddings.npy",
                np.asarray(art["doc_embeddings"], dtype=np.float32))

    if "gate_weights" in art and art["gate_weights"] is not None:
        np.save(art_dir / "gate_weights.npy",
                np.asarray(art["gate_weights"], dtype=np.float32))

    # EC-specific features (expert-choice models only)
    for ec_key in ("affinity_scores", "binary_assignment", "distilled_gate",
                   "expert_embeddings"):
        if ec_key in art and art[ec_key] is not None:
            np.save(art_dir / f"{ec_key}.npy",
                    np.asarray(art[ec_key], dtype=np.float32))

    # WLR-specific: (V, K) vocabulary-level routing matrix
    if "vocab_gate_weights" in art and art["vocab_gate_weights"] is not None:
        np.save(art_dir / "vocab_gate_weights.npy",
                np.asarray(art["vocab_gate_weights"], dtype=np.float32))

    # ------------------------------------------------------------------
    # Human-readable CSVs
    # ------------------------------------------------------------------
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
        if isinstance(dist, (list, tuple)):
            doc_lines.append(str(i) + "," + ",".join(f"{float(p):.6f}" for p in dist))
        else:
            doc_lines.append(str(i))
    save_lines(doc_lines, res_dir / "doc_topic.csv")

    # ------------------------------------------------------------------
    # Evaluate + save metrics
    # # ------------------------------------------------------------------
    # metrics = evaluate_and_save(
    #     method_name, dataset_name, cfg_m, res_dir,
    #     docs=docs, labels=y
    # )
    metrics = evaluate_and_save(
    method_name, dataset_name, cfg_m, res_dir,
    docs=docs_eval, labels=y
)
    metrics.update({
        "time_min":        round(minutes, 3),
        "run_id":          run_id,
        "seed":            seed,
        "stable_model_id": stable_model_id,
    })
    save_json(metrics, res_dir / "metrics.json")

    # ------------------------------------------------------------------
    # Dataset fingerprint — used by analysis_ui to verify that two runs
    # are comparable (same docs, same vocab, etc.) before cross-model
    # comparison. NOT a pure dataset fingerprint: includes model k and
    # vocab_size since both must match for topic-word / theta comparisons
    # to be meaningful.
    # ------------------------------------------------------------------
    fingerprint = {
        "dataset_name":  dataset_name,
        "file_name":     cfg_ds.get("file_name"),
        "min_doc_len":   int(cfg_ds.get("min_doc_len", 0)),
        "labels_flag":   bool(cfg_ds.get("labels", False)),
        "n_docs":        int(len(docs)),
        "vocab_size":    int(len(art["id2word"])),
        "k":             int(len(topics_words)),
        "method":        method_name,
        "seed":          int(seed),
         # NEW: training-time text modifications
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
    ap.add_argument("--dataset_cfg",  required=True)
    ap.add_argument("--method_cfg",   required=True)
    ap.add_argument("--seed",         type=int, default=42)
    ap.add_argument("--results_dir",  default="results_Abstract",
                    help="Root directory for results (default: results_Abstract)")
    args = ap.parse_args()
    run(args.dataset_cfg, args.method_cfg, args.seed, args.results_dir)