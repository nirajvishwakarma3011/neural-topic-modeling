"""
Precompute expert-word associations for all MoE runs.

Implements 3 approaches, saves artifacts/expert_words.json per run.

Approaches:
  1. SenClu-style frequency scoring — distinctive words per expert via
     damped frequency × excess routing probability
  2. Decoder decomposition — push mean topic proportions (t̄_e) through β
     to get a soft word distribution per expert
  3. Expert-topic affinity — t̄_e shows which topics each expert handles;
     read top words from those topics' β rows

Usage:
  python analysis_ui/precompute_expert_words.py          # all MoE runs
  python analysis_ui/precompute_expert_words.py --run results_reuters_10/20260503_164300_moe_ntm_reuters_10
"""

import argparse
import json
import re
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT))


# ── Tokenizer ─────────────────────────────────────────────────────────────────

_STOP = frozenset({
    "the","a","an","and","or","but","in","on","at","to","for","of","with",
    "is","are","was","were","be","been","being","have","has","had","do","does",
    "did","will","would","could","should","may","might","shall","that","this",
    "these","those","it","its","as","by","from","about","into","than","then",
    "so","if","not","no","i","we","you","he","she","they","their","our","your",
    "also","said","says","one","two","three","new","old","up","down","out","over",
    "after","before","between","under","other","more","there","here","when","which",
    "who","what","how","all","any","can","just","only","even","each","some","such",
    "us","my","his","her","its","make","made","use","used","take","get","go","comes",
    "come","per","mln","dlr","dlrs","pct","bln","cts","blns","000","s","re","lt","gt",
})


def tokenize(text: str) -> list[str]:
    tokens = re.findall(r"[a-zA-Z]{3,}", text.lower())
    return [t for t in tokens if t not in _STOP]


# ── Approach 1: SenClu-style frequency scoring ────────────────────────────────

def approach1_senclu(docs: list[str], assignments: np.ndarray, E: int,
                     top_n: int = 20) -> list[dict]:
    """
    score(w|e) = sqrt(max(n(w,e) - n_min, 0)) × (p(e|w) - 1/E)

    n_min = n(w)/E + std_e(n(w,e))
    p(e|w) = n(w,e) / n(w)
    """
    # Build word counts per expert
    expert_counts = [defaultdict(int) for _ in range(E)]
    for idx, doc in enumerate(docs):
        e = int(assignments[idx])
        for word in tokenize(doc):
            expert_counts[e][word] += 1

    # Total counts per word across all experts
    total_counts: dict[str, int] = defaultdict(int)
    for ec in expert_counts:
        for w, c in ec.items():
            total_counts[w] += c

    # Minimum vocab threshold (word must appear ≥ 5 times total)
    vocab = {w for w, c in total_counts.items() if c >= 5}

    results = []
    for e in range(E):
        ec = expert_counts[e]
        scores: dict[str, float] = {}

        for w in vocab:
            n_we = float(ec.get(w, 0))
            n_w = float(total_counts[w])
            # n(w,e) for each expert (for std computation)
            n_across = np.array([float(expert_counts[ex].get(w, 0)) for ex in range(E)])
            std_e = float(n_across.std())

            n_min = n_w / E + std_e
            damped = float(np.sqrt(max(n_we - n_min, 0.0)))
            p_ew = n_we / n_w if n_w > 0 else 0.0
            excess = p_ew - 1.0 / E

            score = damped * excess
            if score > 0:
                scores[w] = score

        top = sorted(scores.items(), key=lambda x: -x[1])[:top_n]
        results.append({
            "expert": e,
            "words": [w for w, _ in top],
            "scores": [round(s, 6) for _, s in top],
        })

    return results


# ── Approach 2: Decoder decomposition ─────────────────────────────────────────

def approach2_decoder(doc_topic: np.ndarray, topic_word_prob: np.ndarray,
                      assignments: np.ndarray, E: int,
                      id2word: list[str], top_n: int = 20) -> list[dict]:
    """
    t̄_e = mean topic proportions for docs where expert e dominates
    word_dist_e = t̄_e @ topic_word_prob   [V]
    Top words by word_dist_e value.
    """
    K, V = topic_word_prob.shape
    results = []

    for e in range(E):
        mask = assignments == e
        if mask.sum() == 0:
            results.append({"expert": e, "words": [], "scores": [],
                            "topic_mix": [], "n_docs": 0})
            continue

        t_mean = doc_topic[mask].mean(axis=0)          # [K]
        word_dist = t_mean @ topic_word_prob             # [V]
        word_dist = word_dist / (word_dist.sum() + 1e-12)

        top_idx = word_dist.argsort()[::-1][:top_n]
        results.append({
            "expert": e,
            "words": [id2word[i] for i in top_idx],
            "scores": [round(float(word_dist[i]), 8) for i in top_idx],
            "topic_mix": [round(float(x), 4) for x in t_mean.tolist()],
            "n_docs": int(mask.sum()),
        })

    return results


# ── Approach 3: Expert-topic affinity ─────────────────────────────────────────

def approach3_topic_affinity(doc_topic: np.ndarray, topic_word_prob: np.ndarray,
                              assignments: np.ndarray, E: int,
                              id2word: list[str], topics_words: list[list[str]],
                              top_n: int = 20, top_topics: int = 3) -> list[dict]:
    """
    t̄_e = mean topic proportions for docs where expert e dominates   [K]
    Top top_topics topics by t̄_e → read words from those β rows.
    Also compute weighted word list: Σ_k  t̄_e[k] * top_n_words(topic_k)
    """
    K, V = topic_word_prob.shape
    results = []

    for e in range(E):
        mask = assignments == e
        if mask.sum() == 0:
            results.append({
                "expert": e, "n_docs": 0,
                "topic_weights": [], "top_topic_ids": [],
                "top_topic_words": [],
                "weighted_words": [], "weighted_scores": [],
            })
            continue

        t_mean = doc_topic[mask].mean(axis=0)       # [K]
        top_k_ids = t_mean.argsort()[::-1][:top_topics].tolist()

        # Weighted word distribution using only top_topics
        top_t = np.zeros(K)
        top_t[top_k_ids] = t_mean[top_k_ids]
        top_t = top_t / (top_t.sum() + 1e-12)

        word_dist = top_t @ topic_word_prob          # [V]
        word_dist = word_dist / (word_dist.sum() + 1e-12)
        top_idx = word_dist.argsort()[::-1][:top_n]

        results.append({
            "expert": e,
            "n_docs": int(mask.sum()),
            "topic_weights": [round(float(t_mean[k]), 4) for k in range(K)],
            "top_topic_ids": top_k_ids,
            "top_topic_words": [topics_words[k] for k in top_k_ids],
            "weighted_words": [id2word[i] for i in top_idx],
            "weighted_scores": [round(float(word_dist[i]), 8) for i in top_idx],
        })

    return results


# ── Runner ────────────────────────────────────────────────────────────────────

def process_run(run_dir: Path) -> bool:
    artifacts = run_dir / "artifacts"
    gate_path = artifacts / "gate_weights.npy"
    if not gate_path.exists():
        print(f"  [skip] no gate_weights: {run_dir.name}")
        return False

    doc_topic_path = artifacts / "doc_topic.npy"
    twp_path = artifacts / "topic_word_prob.npy"
    tw_path = artifacts / "topic_word.npy"
    id2word_path = artifacts / "id2word.json"
    topics_words_path = artifacts / "topics_words.json"
    fp_path = run_dir / "dataset_fingerprint.json"

    if not fp_path.exists():
        print(f"  [skip] no fingerprint: {run_dir.name}")
        return False

    gate = np.load(gate_path)           # [N, E]
    doc_topic = np.load(doc_topic_path) # [N, K]
    N, E = gate.shape
    K = doc_topic.shape[1]

    twp = None
    if twp_path.exists():
        twp = np.load(twp_path)         # [K, V]
    elif tw_path.exists():
        tw = np.load(tw_path)
        twp = tw / (tw.sum(axis=1, keepdims=True) + 1e-12)

    raw_id2word = json.loads(id2word_path.read_text())
    if isinstance(raw_id2word, list):
        id2word_list = raw_id2word
    else:
        id2word_list = [raw_id2word[str(i)] for i in range(len(raw_id2word))]

    topics_words = json.loads(topics_words_path.read_text()) if topics_words_path.exists() else [[] for _ in range(K)]

    fingerprint = json.loads(fp_path.read_text())

    # Load docs via preprocess pipeline
    from src.preprocess import load_dataset
    data_cfg_path = REPO_ROOT / "data_config" / f"{fingerprint['dataset_name']}.json"
    if not data_cfg_path.exists():
        print(f"  [skip] no data_config for {fingerprint['dataset_name']}")
        return False

    cfg = json.loads(data_cfg_path.read_text())
    if "min_doc_len" in fingerprint:
        cfg["min_doc_len"] = fingerprint["min_doc_len"]
    if fingerprint.get("file_name"):
        cfg["file_name"] = fingerprint["file_name"]

    ds = load_dataset(cfg)
    docs = ds.get("eval_docs") or ds["docs"]
    docs = docs[:N]  # safety

    assignments = gate.argmax(axis=1)   # [N]

    print(f"  Approach 1 (SenClu)...")
    ap1 = approach1_senclu(docs, assignments, E, top_n=25)

    ap2, ap3 = [], []
    if twp is not None:
        print(f"  Approach 2 (decoder)...")
        ap2 = approach2_decoder(doc_topic, twp, assignments, E, id2word_list, top_n=25)

        print(f"  Approach 3 (topic affinity)...")
        ap3 = approach3_topic_affinity(doc_topic, twp, assignments, E, id2word_list, topics_words, top_n=25)

    # Expert utilization
    util = [(assignments == e).mean().item() for e in range(E)]

    output = {
        "run_name": run_dir.name,
        "E": E,
        "K": K,
        "N": N,
        "expert_utilization": util,
        "approach1_senclu": ap1,
        "approach2_decoder": ap2,
        "approach3_topic_affinity": ap3,
    }

    out_path = artifacts / "expert_words.json"
    out_path.write_text(json.dumps(output, indent=2))
    print(f"  Saved → {out_path}")
    return True


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--run", default=None, help="Single run dir path")
    args = parser.parse_args()

    if args.run:
        run_dir = Path(args.run)
        if not run_dir.is_absolute():
            run_dir = REPO_ROOT / run_dir
        print(f"Processing: {run_dir.name}")
        process_run(run_dir)
        return

    gate_files = sorted(REPO_ROOT.glob("results_*/*/artifacts/gate_weights.npy"))
    run_dirs = [g.parent.parent for g in gate_files]
    print(f"Found {len(run_dirs)} MoE runs")
    ok = 0
    for rd in run_dirs:
        print(f"\n{rd.name}")
        if process_run(rd):
            ok += 1
    print(f"\nDone: {ok}/{len(run_dirs)} runs processed")


if __name__ == "__main__":
    main()
