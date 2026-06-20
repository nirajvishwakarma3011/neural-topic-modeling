# # src/evaluation/evaluate_models.py
# import json
# import gzip
# import os
# import numpy as np
# from pathlib import Path
# from collections import Counter
# from math import log

# from gensim.corpora import Dictionary
# from gensim.models import CoherenceModel
# from sklearn import metrics as sk_metrics
# from sklearn.metrics import normalized_mutual_info_score

# try:
#     from tqdm import tqdm
# except Exception:
#     def tqdm(x, **kwargs): return x


# # =============================================================================
# # CLUSTERING
# # =============================================================================

# def compute_nmi(true_labels, doc_topic):
#     """
#     Compute Normalized Mutual Information between
#     ground truth labels and predicted topic assignments.
#     """
#     if true_labels is None:
#         return None
#     hard = np.argmax(doc_topic, axis=1)
#     return float(normalized_mutual_info_score(true_labels, hard))


# def purity_score(y_true, y_pred):
#     contingency = sk_metrics.cluster.contingency_matrix(y_true, y_pred)
#     return np.sum(np.amax(contingency, axis=0)) / np.sum(contingency)


# def evaluate_clustering(theta: np.ndarray, labels: np.ndarray):
#     preds = np.argmax(theta, axis=1)
#     return {
#         "purity": float(purity_score(labels, preds)),
#         "nmi_sklearn": float(sk_metrics.cluster.normalized_mutual_info_score(labels, preds)),
#     }


# # =============================================================================
# # COHERENCE
# # =============================================================================

# def compute_pmi_from_paper(topics_top_words, docs, topk=10):
#     """
#     Compute topic coherence (NPMI) based on document-wise co-occurrence.
#     """
#     # Collect all words we need to track across all topics
#     words_to_track = set()
#     for topic in topics_top_words:
#         for word in topic[:topk]:
#             words_to_track.add(word)

#     word_doc_counts = Counter()
#     pair_doc_counts = Counter()
#     total_docs = len(docs)

#     # Count per-document occurrences and co-occurrences
#     for doc in tqdm(docs, desc="Counting document co-occurrences"):
#         unique_words_in_doc = words_to_track.intersection(doc.split())

#         if len(unique_words_in_doc) < 2:
#             for word in unique_words_in_doc:
#                 word_doc_counts[word] += 1
#             continue

#         for word in unique_words_in_doc:
#             word_doc_counts[word] += 1

#         doc_words_list = list(unique_words_in_doc)
#         for i in range(len(doc_words_list)):
#             for j in range(i + 1, len(doc_words_list)):
#                 w1, w2 = doc_words_list[i], doc_words_list[j]
#                 pair = tuple(sorted((w1, w2)))
#                 pair_doc_counts[pair] += 1

#     # Compute NPMI for every word pair in each topic
#     topic_scores = []
#     for topic in topics_top_words:
#         words = topic[:topk]
#         pair_scores = []
#         for i in range(len(words)):
#             for j in range(i + 1, len(words)):
#                 w1, w2 = words[i], words[j]
#                 pair = tuple(sorted((w1, w2)))

#                 p_w1   = word_doc_counts[w1]   / total_docs
#                 p_w2   = word_doc_counts[w2]   / total_docs
#                 p_w1_w2 = pair_doc_counts[pair] / total_docs

#                 if p_w1_w2 == 0 or p_w1 == 0 or p_w2 == 0:
#                     continue

#                 pmi  = log(p_w1_w2 / (p_w1 * p_w2) + 1e-12)
#                 npmi = pmi / (-log(p_w1_w2 + 1e-12))
#                 pair_scores.append(npmi)

#         if pair_scores:
#             topic_scores.append(np.median(pair_scores))   # median per topic

#     return float(np.mean(topic_scores)) if topic_scores else 0.0


# def compute_topic_diversity(topics_words):
#     """
#     Standard TD: fraction of unique words among all top words.
#     topics_words: List[List[str]]
#     """
#     all_words = [w for topic in topics_words for w in topic]
#     if not all_words:
#         return 0.0
#     return len(set(all_words)) / len(all_words)


# def _simple_tokenize(text: str):
#     return [w for w in text.lower().split() if w]


# def compute_cv_coherence(topics_words, reference_texts, coherence_type="c_v", max_docs=20000):
#     """
#     Gensim CoherenceModel (c_v / u_mass / c_uci / c_npmi).
#     reference_texts: List[str] raw texts
#     """
#     if not reference_texts:
#         return None

#     ref = reference_texts[:max_docs]
#     tokenized = [_simple_tokenize(t) for t in ref]
#     tokenized = [toks for toks in tokenized if toks]

#     dictionary = Dictionary(tokenized)
#     cm = CoherenceModel(
#         topics=topics_words,
#         texts=tokenized,
#         dictionary=dictionary,
#         coherence=coherence_type,
#     )
#     return float(cm.get_coherence())


# # =============================================================================
# # PALMETTO / CV via Java
# # =============================================================================

# def write_topics_for_palmetto(topics_words, out_path):
#     with open(out_path, "w") as f:
#         for topic in topics_words:
#             f.write(" ".join(topic) + "\n")


# def TC_on_wikipedia(top_word_path, cv_type="C_V"):
#     """
#     Compute the TC score on the Wikipedia dataset via Palmetto Java tool.
#     """
#     jar_dir  = "/data4/home/nirajv/tm_benchmark_json/data"
#     wiki_dir = os.path.join(".", "/data4/home/nirajv/tm_benchmark_json/data")
#     random_number = np.random.randint(100000)
#     os.system(
#         f"java -jar {os.path.join(jar_dir, 'pametto.jar')} "
#         f"{os.path.join(wiki_dir, 'wikipedia', 'wikipedia_bd')} "
#         f"{cv_type} {top_word_path} > tmp{random_number}.txt"
#     )
#     cv_score = []
#     with open(f"tmp{random_number}.txt", "r") as f:
#         for line in f.readlines():
#             if not line.startswith("202"):
#                 cv_score.append(float(line.strip().split()[1]))
#     os.remove(f"tmp{random_number}.txt")
#     return cv_score, sum(cv_score) / len(cv_score)


# # =============================================================================
# # WIKIPEDIA CORPUS LOADER
# # =============================================================================

# def _read_wiki_docs_gz(path: Path, max_lines: int | None = None):
#     lines = []
#     with gzip.open(path, "rt", encoding="utf-8", errors="ignore") as f:
#         for i, line in enumerate(f):
#             if max_lines is not None and i >= max_lines:
#                 break
#             lines.append(line.rstrip("\n"))
#     return lines


# # =============================================================================
# # MAIN ENTRY POINT
# # =============================================================================

# def evaluate_and_save(
#     method_name,
#     dataset_name,
#     cfg_method,
#     out_dir,
#     docs=None,
#     labels=None,
# ):
#     out_dir = Path(out_dir)
#     out_dir.mkdir(parents=True, exist_ok=True)

#     # --- Load artifacts saved by main.py ---
#     art_dir     = out_dir / "artifacts"
#     topics_path = art_dir / "topics_words.json"
#     theta_path  = art_dir / "doc_topic.npy"

#     if not topics_path.exists():
#         raise FileNotFoundError(f"Missing {topics_path}. Ensure artifacts were saved in main.py.")
#     if not theta_path.exists():
#         raise FileNotFoundError(f"Missing {theta_path}. Ensure artifacts were saved in main.py.")

#     topics_top_words = json.loads(topics_path.read_text())
#     doc_topic        = np.load(theta_path)

#     # --- NPMI on Wikipedia --- 
#     wiki_gz = Path("data/wiki_docs_100k.txt.gz")
#     # wiki_gz = Path("/data4/home/nirajv/tm_benchmark_json/data/raw2/wiki_docs_full.txt.gz")
    

#     if wiki_gz.exists():
#         wiki_docs  = _read_wiki_docs_gz(wiki_gz, max_lines=None)
#         npmi_score = compute_pmi_from_paper(topics_top_words, wiki_docs, topk=10)
#     else:
#         wiki_docs  = []
#         npmi_score = -1
#         print("[warn] Wikipedia file not found; npmi set to -1.")

#     # --- Topic Diversity ---
#     td_score = compute_topic_diversity(topics_top_words)

#     # --- CV coherence via Palmetto ---
#     tmp_topics = out_dir / "topics_for_palmetto.txt"
#     write_topics_for_palmetto(topics_top_words, tmp_topics)
#     cv_per_topic, cv_score = TC_on_wikipedia(str(tmp_topics))

#     # --- Clustering ---
#     nmi_score        = compute_nmi(labels, np.array(doc_topic)) if labels is not None else None
#     clustering_extra = evaluate_clustering(np.asarray(doc_topic), np.asarray(labels)) if labels is not None else None

#     # --- Assemble metrics ---
#     metrics_out = {
#         "dataset": dataset_name,
#         "method":  method_name,
#         "k":       cfg_method.get("k", len(topics_top_words)),

#         # coherence
#         "npmi_paper": round(float(npmi_score), 4),
#         "cv":         round(float(cv_score), 4) if cv_score is not None else None,

#         # diversity
#         "topic_diversity": round(float(td_score), 6),

#         # clustering
#         "nmi":        round(float(nmi_score), 4)                      if nmi_score        is not None else None,
#         "purity":     round(float(clustering_extra["purity"]), 4)     if clustering_extra is not None else None,
#         # "nmi_sklearn":round(float(clustering_extra["nmi_sklearn"]), 4) if clustering_extra is not None else None,
#     }

#     return metrics_out


##### This is stramlit version - NPMI is per topic 

# src/evaluation/evaluate_models.py
import json
import gzip
import os
import numpy as np
from pathlib import Path
from collections import Counter
from math import log

from gensim.corpora import Dictionary
from gensim.models import CoherenceModel
from sklearn import metrics as sk_metrics
from sklearn.metrics import normalized_mutual_info_score

try:
    from tqdm import tqdm
except Exception:
    def tqdm(x, **kwargs): return x


# =============================================================================
# CLUSTERING
# =============================================================================

def compute_nmi(true_labels, doc_topic):
    """
    Compute Normalized Mutual Information between
    ground truth labels and predicted topic assignments.
    """
    if true_labels is None:
        return None
    hard = np.argmax(doc_topic, axis=1)
    return float(normalized_mutual_info_score(true_labels, hard))


def purity_score(y_true, y_pred):
    contingency = sk_metrics.cluster.contingency_matrix(y_true, y_pred)
    return np.sum(np.amax(contingency, axis=0)) / np.sum(contingency)


def evaluate_clustering(theta: np.ndarray, labels: np.ndarray):
    preds = np.argmax(theta, axis=1)
    return {
        "purity": float(purity_score(labels, preds)),
        "nmi_sklearn": float(sk_metrics.cluster.normalized_mutual_info_score(labels, preds)),
    }


# =============================================================================
# COHERENCE
# =============================================================================

def compute_pmi_from_paper(topics_top_words, docs, topk=10):
    """
    Compute topic coherence (NPMI) based on document-wise co-occurrence.
    """
    # Collect all words we need to track across all topics
    words_to_track = set()
    for topic in topics_top_words:
        for word in topic[:topk]:
            words_to_track.add(word)

    word_doc_counts = Counter()
    pair_doc_counts = Counter()
    total_docs = len(docs)

    # Count per-document occurrences and co-occurrences
    for doc in tqdm(docs, desc="Counting document co-occurrences"):
        unique_words_in_doc = words_to_track.intersection(doc.split())

        if len(unique_words_in_doc) < 2:
            for word in unique_words_in_doc:
                word_doc_counts[word] += 1
            continue

        for word in unique_words_in_doc:
            word_doc_counts[word] += 1

        doc_words_list = list(unique_words_in_doc)
        for i in range(len(doc_words_list)):
            for j in range(i + 1, len(doc_words_list)):
                w1, w2 = doc_words_list[i], doc_words_list[j]
                pair = tuple(sorted((w1, w2)))
                pair_doc_counts[pair] += 1

    # Compute NPMI for every word pair in each topic
    topic_scores = []
    for topic in topics_top_words:
        words = topic[:topk]
        pair_scores = []
        for i in range(len(words)):
            for j in range(i + 1, len(words)):
                w1, w2 = words[i], words[j]
                pair = tuple(sorted((w1, w2)))

                p_w1   = word_doc_counts[w1]   / total_docs
                p_w2   = word_doc_counts[w2]   / total_docs
                p_w1_w2 = pair_doc_counts[pair] / total_docs

                if p_w1_w2 == 0 or p_w1 == 0 or p_w2 == 0:
                    continue

                pmi  = log(p_w1_w2 / (p_w1 * p_w2) + 1e-12)
                npmi = pmi / (-log(p_w1_w2 + 1e-12))
                pair_scores.append(npmi)

        if pair_scores:
            topic_scores.append(float(np.median(pair_scores)))   # median per topic
        else:
            topic_scores.append(0.0)  # keep alignment with topic index

    mean_npmi = float(np.mean(topic_scores)) if topic_scores else 0.0
    return mean_npmi, topic_scores


def compute_topic_diversity(topics_words):
    """
    Standard TD: fraction of unique words among all top words.
    topics_words: List[List[str]]
    """
    all_words = [w for topic in topics_words for w in topic]
    if not all_words:
        return 0.0
    return len(set(all_words)) / len(all_words)


def _simple_tokenize(text: str):
    return [w for w in text.lower().split() if w]


def compute_cv_coherence(topics_words, reference_texts, coherence_type="c_v", max_docs=20000):
    """
    Gensim CoherenceModel (c_v / u_mass / c_uci / c_npmi).
    reference_texts: List[str] raw texts
    """
    if not reference_texts:
        return None

    ref = reference_texts[:max_docs]
    tokenized = [_simple_tokenize(t) for t in ref]
    tokenized = [toks for toks in tokenized if toks]

    dictionary = Dictionary(tokenized)
    cm = CoherenceModel(
        topics=topics_words,
        texts=tokenized,
        dictionary=dictionary,
        coherence=coherence_type,
    )
    return float(cm.get_coherence())


# =============================================================================
# PALMETTO / CV via Java
# =============================================================================

def write_topics_for_palmetto(topics_words, out_path):
    with open(out_path, "w") as f:
        for topic in topics_words:
            f.write(" ".join(topic) + "\n")


def TC_on_wikipedia(top_word_path, cv_type="C_V"):
    """
    Compute the TC score on the Wikipedia dataset via Palmetto Java tool.
    """
    jar_dir  = "/data4/home/nirajv/tm_benchmark_json/data"
    wiki_dir = os.path.join(".", "/data4/home/nirajv/tm_benchmark_json/data")
    random_number = np.random.randint(100000)
    os.system(
        f"java -jar {os.path.join(jar_dir, 'pametto.jar')} "
        f"{os.path.join(wiki_dir, 'wikipedia', 'wikipedia_bd')} "
        f"{cv_type} {top_word_path} > tmp{random_number}.txt"
    )
    cv_score = []
    with open(f"tmp{random_number}.txt", "r") as f:
        for line in f.readlines():
            if not line.startswith("202"):
                cv_score.append(float(line.strip().split()[1]))
    os.remove(f"tmp{random_number}.txt")
    return cv_score, sum(cv_score) / len(cv_score)


# =============================================================================
# WIKIPEDIA CORPUS LOADER
# =============================================================================

def _read_wiki_docs_gz(path: Path, max_lines: int | None = None):
    lines = []
    with gzip.open(path, "rt", encoding="utf-8", errors="ignore") as f:
        for i, line in enumerate(f):
            if max_lines is not None and i >= max_lines:
                break
            lines.append(line.rstrip("\n"))
    return lines


# =============================================================================
# MAIN ENTRY POINT
# =============================================================================

def evaluate_and_save(
    method_name,
    dataset_name,
    cfg_method,
    out_dir,
    docs=None,
    labels=None,
):
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    # --- Load artifacts saved by main.py ---
    art_dir     = out_dir / "artifacts"
    topics_path = art_dir / "topics_words.json"
    theta_path  = art_dir / "doc_topic.npy"

    if not topics_path.exists():
        raise FileNotFoundError(f"Missing {topics_path}. Ensure artifacts were saved in main.py.")
    if not theta_path.exists():
        raise FileNotFoundError(f"Missing {theta_path}. Ensure artifacts were saved in main.py.")

    topics_top_words = json.loads(topics_path.read_text())
    doc_topic        = np.load(theta_path)

    # --- NPMI on Wikipedia --- 
    wiki_gz = Path("data/wiki_docs_100k.txt.gz")
    # wiki_gz = Path("/data4/home/nirajv/tm_benchmark_json/data/raw2/wiki_docs_full.txt.gz")
    

    if wiki_gz.exists():
        wiki_docs  = _read_wiki_docs_gz(wiki_gz, max_lines=None)
        npmi_score, npmi_per_topic = compute_pmi_from_paper(topics_top_words, wiki_docs, topk=10)
    else:
        wiki_docs       = []
        npmi_score      = -1
        npmi_per_topic  = []
        print("[warn] Wikipedia file not found; npmi set to -1.")

    # Save per-topic NPMI as a separate artifact for the analysis UI.
    # Aligned by topic index. Empty list if wiki was missing.
    with open(out_dir / "npmi_per_topic.json", "w") as _f:
        json.dump(npmi_per_topic, _f)

    # --- Topic Diversity ---
    td_score = compute_topic_diversity(topics_top_words)

    # --- CV coherence via Palmetto ---
    tmp_topics = out_dir / "topics_for_palmetto.txt"
    write_topics_for_palmetto(topics_top_words, tmp_topics)
    cv_per_topic, cv_score = TC_on_wikipedia(str(tmp_topics))

    # --- Clustering ---
    nmi_score        = compute_nmi(labels, np.array(doc_topic)) if labels is not None else None
    clustering_extra = evaluate_clustering(np.asarray(doc_topic), np.asarray(labels)) if labels is not None else None

    # --- Assemble metrics ---
    metrics_out = {
        "dataset": dataset_name,
        "method":  method_name,
        "k":       cfg_method.get("k", len(topics_top_words)),

        # coherence
        "npmi_paper": round(float(npmi_score), 4),
        "cv":         round(float(cv_score), 4) if cv_score is not None else None,

        # diversity
        "topic_diversity": round(float(td_score), 6),

        # clustering
        "nmi":        round(float(nmi_score), 4)                      if nmi_score        is not None else None,
        "purity":     round(float(clustering_extra["purity"]), 4)     if clustering_extra is not None else None,
        # "nmi_sklearn":round(float(clustering_extra["nmi_sklearn"]), 4) if clustering_extra is not None else None,
    }

    return metrics_out