# analyse_vae.py
#
# Two jobs:
#   1. Plot training loss curves from training_log.csv
#   2. Analyse topic vectors, word vectors, and doc-topic distributions
#
# Usage:
#   python analyse_vae.py --run_dir results_fresh_StackOverflow/<run_id>
#
# Requires the model to have been saved so we can reload it for vector analysis.
# All plots saved to <run_dir>/analysis/

# python analyse_vae.py --run_dir /data4/home/nirajv/small_text/results_Abstract/20260424_132552_vae_gsm_abstract

import argparse
import json
import csv
import sys
import numpy as np
import matplotlib
matplotlib.use("Agg")   # no display needed
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
from scipy.spatial.distance import pdist, squareform
from scipy.stats import entropy as scipy_entropy


# =============================================================================
# 1. TRAINING CURVES
# =============================================================================

def plot_training_curves(log_path: Path, out_dir: Path):
    """
    Reads training_log.csv and plots:
      • loss / recon / kl over epochs         → loss_curves.png
      • kl_weight schedule                    → kl_schedule.png
      • unique_top_words over epochs          → topic_collapse.png
    """
    if not log_path.exists():
        print(f"[warn] {log_path} not found — skipping training curves.")
        return

    epochs, loss, recon, kl, div, kl_w, unique = [], [], [], [], [], [], []
    with open(log_path) as f:
        for row in csv.DictReader(f):
            epochs.append(int(row["epoch"]))
            loss.append(float(row["loss"]))
            recon.append(float(row["recon"]))
            kl.append(float(row["kl"]))
            div.append(float(row["div"]))
            kl_w.append(float(row["kl_weight"]))
            unique.append(int(row["unique_top_words"]))

    epochs = np.array(epochs)

    # ── Plot 1: Loss decomposition ──────────────────────────────────────────
    fig, axes = plt.subplots(1, 3, figsize=(14, 4))

    axes[0].plot(epochs, loss,  label="total loss",    color="black")
    axes[0].set_title("Total Loss")
    axes[0].set_xlabel("Epoch")

    axes[1].plot(epochs, recon, label="reconstruction", color="steelblue")
    axes[1].set_title("Reconstruction Loss\n(higher = better recon)")
    axes[1].set_xlabel("Epoch")

    axes[2].plot(epochs, kl,   label="KL",             color="tomato")
    axes[2].set_title("KL Divergence\n(~0 = posterior collapse)")
    axes[2].set_xlabel("Epoch")

    for ax in axes:
        ax.grid(True, alpha=0.3)
        ax.legend()

    plt.tight_layout()
    plt.savefig(out_dir / "loss_curves.png", dpi=150)
    plt.close()
    print(f"  saved loss_curves.png")

    # ── Plot 2: KL weight schedule ──────────────────────────────────────────
    fig, ax1 = plt.subplots(figsize=(8, 4))
    ax2 = ax1.twinx()

    ax1.plot(epochs, kl_w, color="orange", label="kl_weight")
    ax2.plot(epochs, kl,   color="tomato",  label="kl value", alpha=0.6)

    ax1.set_xlabel("Epoch")
    ax1.set_ylabel("KL weight (0→1)", color="orange")
    ax2.set_ylabel("KL value",         color="tomato")
    ax1.set_title("KL Annealing Schedule vs Actual KL")

    lines1, labels1 = ax1.get_legend_handles_labels()
    lines2, labels2 = ax2.get_legend_handles_labels()
    ax1.legend(lines1 + lines2, labels1 + labels2, loc="upper left")
    ax1.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(out_dir / "kl_schedule.png", dpi=150)
    plt.close()
    print(f"  saved kl_schedule.png")

    # ── Plot 3: Topic collapse diagnostic ───────────────────────────────────
    K = max(unique)
    fig, ax = plt.subplots(figsize=(8, 4))
    ax.plot(epochs, unique, color="green")
    ax.axhline(K, color="gray", linestyle="--", label=f"K={K} (no collapse)")
    ax.fill_between(epochs, unique, K, alpha=0.1, color="red",
                    label="collapsed topics")
    ax.set_xlabel("Epoch")
    ax.set_ylabel("Topics with unique dominant word")
    ax.set_title("Topic Collapse Diagnostic\n(lower = more collapsed)")
    ax.legend()
    ax.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(out_dir / "topic_collapse.png", dpi=150)
    plt.close()
    print(f"  saved topic_collapse.png")


# =============================================================================
# 2. TOPIC VECTOR ANALYSIS
# =============================================================================

def plot_topic_similarity(topic_vectors: np.ndarray, topics_words: list, out_dir: Path):
    """
    Cosine similarity matrix between all topic pairs.
    High off-diagonal values = redundant topics.
    """
    # Normalize to unit vectors
    norms  = np.linalg.norm(topic_vectors, axis=1, keepdims=True)
    normed = topic_vectors / (norms + 1e-10)
    sim    = normed @ normed.T                         # (K, K) cosine sim

    # Short labels for axes
    labels = [f"T{i}: {' '.join(w[:2])}" for i, w in enumerate(topics_words)]

    fig, ax = plt.subplots(figsize=(max(6, len(labels)*0.6), max(5, len(labels)*0.55)))
    sns.heatmap(sim, annot=True, fmt=".2f", cmap="coolwarm",
                xticklabels=labels, yticklabels=labels,
                vmin=-1, vmax=1, ax=ax)
    ax.set_title("Topic-Topic Cosine Similarity\n(off-diagonal≈1 means redundant topics)")
    plt.tight_layout()
    plt.savefig(out_dir / "topic_similarity.png", dpi=150)
    plt.close()
    print(f"  saved topic_similarity.png")


def plot_topic_entropy(beta: np.ndarray, topics_words: list, out_dir: Path):
    """
    Entropy of each topic's word distribution β_k.
    Low entropy = sharp/focused topic.
    High entropy = diffuse/garbage topic.
    """
    entropies = [scipy_entropy(beta[k]) for k in range(len(beta))]
    labels    = [f"T{i}: {topics_words[i][0]}" for i in range(len(beta))]

    fig, ax = plt.subplots(figsize=(max(6, len(labels)*0.7), 4))
    bars = ax.bar(labels, entropies, color="steelblue", edgecolor="black")
    ax.axhline(np.mean(entropies), color="red", linestyle="--",
               label=f"mean={np.mean(entropies):.2f}")
    ax.set_title("Topic Entropy\n(lower = more focused topic)")
    ax.set_ylabel("Entropy")
    ax.set_xlabel("Topic")
    plt.xticks(rotation=45, ha="right")
    ax.legend()
    plt.tight_layout()
    plt.savefig(out_dir / "topic_entropy.png", dpi=150)
    plt.close()
    print(f"  saved topic_entropy.png")

    # Also save as CSV for further analysis
    with open(out_dir / "topic_entropy.csv", "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["topic_id", "top_word", "entropy"])
        for i, (e, lbl) in enumerate(zip(entropies, labels)):
            w.writerow([i, topics_words[i][0], round(e, 6)])
    print(f"  saved topic_entropy.csv")


# =============================================================================
# 3. WORD VECTOR ANALYSIS
# =============================================================================

def nearest_word_neighbors(word_vectors: np.ndarray, vocab: list,
                            query_words: list, topn: int = 10, out_dir: Path = None):
    """
    For each query word, find its nearest neighbors in word embedding space.
    Answers: "what words does the model consider semantically similar?"

    word_vectors: (V, H)
    vocab:        [str] length V
    query_words:  words to query — if not in vocab, skipped
    """
    vocab_set = {w: i for i, w in enumerate(vocab)}
    norms     = np.linalg.norm(word_vectors, axis=1, keepdims=True)
    normed    = word_vectors / (norms + 1e-10)

    results = {}
    for qw in query_words:
        if qw not in vocab_set:
            print(f"  [warn] '{qw}' not in vocabulary, skipping.")
            continue
        idx  = vocab_set[qw]
        sims = normed @ normed[idx]            # cosine sim to all words
        sims[idx] = -1                         # exclude self
        top_idx = np.argsort(-sims)[:topn]
        neighbors = [(vocab[i], round(float(sims[i]), 4)) for i in top_idx]
        results[qw] = neighbors
        print(f"  Nearest to '{qw}': {[w for w, _ in neighbors[:5]]}")

    if out_dir and results:
        with open(out_dir / "word_neighbors.json", "w") as f:
            json.dump(results, f, indent=2)
        print(f"  saved word_neighbors.json")

    return results


def plot_word_topic_affinity(beta: np.ndarray, vocab: list,
                             topics_words: list, out_dir: Path):
    """
    For each topic's top words, show how exclusively those words belong to that
    topic vs being shared across topics.

    beta[:, w] = probability of word w across all topics
    If beta[k, w] dominates → word is topic-specific
    If beta[:, w] is uniform → word is generic/shared
    """
    K    = beta.shape[0]
    topn = 8

    fig, axes = plt.subplots(2, (K + 1) // 2,
                             figsize=(min(20, K * 2), 6))
    axes = axes.flatten()

    for k in range(K):
        top_idx   = np.argsort(-beta[k])[:topn]
        top_words = [vocab[i] for i in top_idx]
        # For each top word: what fraction of its total probability is in topic k?
        # beta[:, i].sum() = total weight of word i across all topics
        exclusivity = [
            beta[k, i] / (beta[:, i].sum() + 1e-10)
            for i in top_idx
        ]
        colors = ["green" if e > 0.5 else "orange" if e > 0.2 else "red"
                  for e in exclusivity]
        axes[k].barh(top_words[::-1], exclusivity[::-1], color=colors[::-1])
        axes[k].set_xlim(0, 1)
        axes[k].set_title(f"T{k}: exclusivity")
        axes[k].axvline(0.5, color="black", linestyle="--", linewidth=0.8)

    # hide unused subplots
    for j in range(K, len(axes)):
        axes[j].set_visible(False)

    plt.suptitle("Word Exclusivity per Topic\n"
                 "(green>0.5: word belongs mostly to this topic, "
                 "red<0.2: word is shared)", fontsize=10)
    plt.tight_layout()
    plt.savefig(out_dir / "word_topic_exclusivity.png", dpi=150)
    plt.close()
    print(f"  saved word_topic_exclusivity.png")


# =============================================================================
# 4. DOCUMENT-TOPIC ANALYSIS
# =============================================================================

def plot_doc_topic_analysis(doc_topic: np.ndarray, out_dir: Path):
    """
    Three views of the doc-topic matrix:
      • Distribution of dominant topic probability (model confidence)
      • Document entropy histogram (how mixed each doc is)
      • Topic load bar chart (how many docs each topic captures)
    """
    N, K = doc_topic.shape

    confidence    = np.max(doc_topic, axis=1)          # per-doc max prob
    doc_entropies = scipy_entropy(doc_topic.T).T        # per-doc entropy — (N,)
    topic_load    = np.argmax(doc_topic, axis=1)        # dominant topic per doc
    load_counts   = np.bincount(topic_load, minlength=K)

    fig, axes = plt.subplots(1, 3, figsize=(15, 4))

    # ── Confidence histogram ─────────────────────────────────────────────────
    axes[0].hist(confidence, bins=30, color="steelblue", edgecolor="black")
    axes[0].axvline(confidence.mean(), color="red", linestyle="--",
                    label=f"mean={confidence.mean():.2f}")
    axes[0].set_title("Document Confidence\n(max topic prob per doc)")
    axes[0].set_xlabel("Max topic probability")
    axes[0].set_ylabel("# documents")
    axes[0].legend()

    # ── Entropy histogram ─────────────────────────────────────────────────────
    axes[1].hist(doc_entropies, bins=30, color="salmon", edgecolor="black")
    axes[1].axvline(doc_entropies.mean(), color="darkred", linestyle="--",
                    label=f"mean={doc_entropies.mean():.2f}")
    axes[1].set_title("Document Entropy\n(higher = more topic-mixed doc)")
    axes[1].set_xlabel("Entropy")
    axes[1].set_ylabel("# documents")
    axes[1].legend()

    # ── Topic load bar chart ──────────────────────────────────────────────────
    axes[2].bar(range(K), load_counts, color="mediumseagreen", edgecolor="black")
    axes[2].axhline(N / K, color="red", linestyle="--",
                    label=f"uniform={N//K} docs/topic")
    axes[2].set_title("Topic Load\n(# docs assigned to each topic)")
    axes[2].set_xlabel("Topic ID")
    axes[2].set_ylabel("# documents")
    axes[2].legend()

    plt.tight_layout()
    plt.savefig(out_dir / "doc_topic_analysis.png", dpi=150)
    plt.close()
    print(f"  saved doc_topic_analysis.png")

    # Save per-doc stats as CSV for further analysis
    with open(out_dir / "doc_stats.csv", "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["doc_id", "dominant_topic", "confidence", "entropy"])
        for i in range(N):
            w.writerow([i, int(topic_load[i]),
                        round(float(confidence[i]), 6),
                        round(float(doc_entropies[i]), 6)])
    print(f"  saved doc_stats.csv  ({N} rows)")




# =============================================================================
# 5. TSNE PLOTS FROM CSV — original cluster vs predicted topic
# =============================================================================

def plot_tsne_from_csv(data_csv: Path, doc_topic: np.ndarray,
                       topics_words: list, out_dir: Path):
    """
    Reads a CSV with columns: text, cluster, tsne_x, tsne_y
    Produces two side-by-side t-SNE plots:
      Left:  documents coloured by original ground-truth cluster
      Right: same documents coloured by VAE-GSM predicted topic

    The two plots share the same axes so positions are directly comparable.
    """
    if not data_csv.exists():
        print(f"  [skip] {data_csv} not found")
        return

    import pandas as pd
    df = pd.read_csv(data_csv)

    required = {"cluster", "tsne_x", "tsne_y"}
    if not required.issubset(df.columns):
        print(f"  [skip] CSV missing columns {required - set(df.columns)}")
        return

    N_csv   = len(df)
    N_model = len(doc_topic)
    if N_csv != N_model:
        print(f"  [warn] CSV has {N_csv} rows but doc_topic has {N_model} rows — "
              f"using first {min(N_csv, N_model)} rows")
        n = min(N_csv, N_model)
        df        = df.iloc[:n].reset_index(drop=True)
        doc_topic = doc_topic[:n]

    x = df["tsne_x"].values
    y = df["tsne_y"].values

    # ── Ground-truth cluster colours ────────────────────────────────────────
    orig_clusters  = df["cluster"].values
    unique_clusters = sorted(df["cluster"].unique())
    n_orig          = len(unique_clusters)
    orig_palette    = plt.cm.get_cmap("tab20", n_orig)
    orig_cmap       = {c: orig_palette(i) for i, c in enumerate(unique_clusters)}

    # ── Predicted topic colours ──────────────────────────────────────────────
    pred_topics    = np.argmax(doc_topic, axis=1)
    K              = doc_topic.shape[1]
    pred_palette   = plt.cm.get_cmap("tab10", K)
    pred_cmap      = {k: pred_palette(k) for k in range(K)}

    fig, axes = plt.subplots(1, 2, figsize=(20, 9))
    point_size = max(2, min(6, 30000 / len(df)))   # scale dot size to N

    # ── Left: original clusters ──────────────────────────────────────────────
    ax = axes[0]
    for c in unique_clusters:
        mask = orig_clusters == c
        ax.scatter(x[mask], y[mask], s=point_size,
                   color=orig_cmap[c], label=str(c), alpha=0.7, linewidths=0)
    ax.set_title("t-SNE — Original Cluster Labels", fontsize=13, fontweight="bold")
    ax.set_xlabel("tsne_x")
    ax.set_ylabel("tsne_y")
    # Legend: show cluster IDs, limit to 25 to avoid overflow
    handles, labels = ax.get_legend_handles_labels()
    if len(handles) <= 25:
        ax.legend(handles, labels, title="Cluster", markerscale=3,
                  fontsize=7, loc="best", ncol=max(1, n_orig // 15))
    else:
        ax.legend(handles[:25], labels[:25], title="Cluster (first 25)",
                  markerscale=3, fontsize=7, loc="best", ncol=2)
    ax.grid(True, alpha=0.2)

    # ── Right: predicted topics ──────────────────────────────────────────────
    ax = axes[1]
    for k in range(K):
        mask  = pred_topics == k
        label = f"T{k}: {topics_words[k][0]}"   # dominant word as label
        ax.scatter(x[mask], y[mask], s=point_size,
                   color=pred_cmap[k], label=label, alpha=0.7, linewidths=0)

    ax.set_title("t-SNE — Predicted Topic (VAE-GSM)", fontsize=13, fontweight="bold")
    ax.set_xlabel("tsne_x")
    ax.set_ylabel("tsne_y")
    ax.legend(title="Topic", markerscale=3, fontsize=8, loc="best",
              ncol=max(1, K // 8))
    ax.grid(True, alpha=0.2)

    plt.tight_layout()
    plt.savefig(out_dir / "tsne_cluster_vs_topic.png", dpi=160, bbox_inches="tight")
    plt.close()
    print(f"  saved tsne_cluster_vs_topic.png  ({len(df)} points)")


# =============================================================================
# 6. WORD + TOPIC VECTOR SPACE PLOT
# =============================================================================

def plot_word_topic_space(word_vectors: np.ndarray, topic_vectors: np.ndarray,
                          vocab: list, topics_words: list,
                          beta: np.ndarray, out_dir: Path,
                          topn_words: int = 15, perplexity: int = 40):
    """
    Projects word vectors AND topic vectors into the same 2D space using t-SNE,
    then colours each word by its dominant topic.

    What this shows:
      - Words that co-occur in topics cluster together
      - Topic centroids sit near their member words
      - Topics that share vocabulary appear closer together
      - Allows direct visual inspection of "does the model understand
        that 'haskell' and 'type' belong together?"

    Strategy:
      1. Pick the top-N words per topic by β probability
      2. Collect their word vectors
      3. Concatenate with topic vectors
      4. Run t-SNE on the combined matrix
      5. Plot words (small dots) + topic labels (large stars)

    topn_words: how many top words per topic to include in the plot
    """
    from sklearn.manifold import TSNE

    K = len(topics_words)
    V, H = word_vectors.shape

    # ── Select top words per topic ───────────────────────────────────────────
    selected_word_ids  = []   # index into vocab
    word_topic_labels  = []   # which topic owns this word (dominant)

    for k in range(K):
        top_idx = np.argsort(-beta[k])[:topn_words]
        for idx in top_idx:
            if idx not in selected_word_ids:          # deduplicate shared words
                selected_word_ids.append(idx)
                # dominant topic for this word = argmax over beta[:, idx]
                word_topic_labels.append(int(np.argmax(beta[:, idx])))

    selected_word_vecs = word_vectors[selected_word_ids]   # (W, H)
    selected_words     = [vocab[i] for i in selected_word_ids]

    # ── Build combined matrix: [word vecs | topic vecs] ─────────────────────
    # Topic vectors are in the same H-dim space as word vectors (both are t and v
    # in the paper), so t-SNE can project them jointly.
    combined = np.vstack([selected_word_vecs, topic_vectors])   # (W+K, H)
    n_words  = len(selected_word_ids)

    # ── t-SNE ────────────────────────────────────────────────────────────────
    n_total = combined.shape[0]
    perp    = min(perplexity, n_total // 3)   # perplexity < n_samples/3
    print(f"  running t-SNE on {n_total} points (words={n_words}, topics={K}) ...")
    tsne   = TSNE(n_components=2, perplexity=perp, max_iter=1000,
                  random_state=42, init="pca")
    coords = tsne.fit_transform(combined)     # (W+K, 2)

    word_coords  = coords[:n_words]           # (W, 2)
    topic_coords = coords[n_words:]           # (K, 2)

    # ── Colour palette ───────────────────────────────────────────────────────
    palette = plt.cm.get_cmap("tab10", K)
    colors  = [palette(k) for k in range(K)]

    # ── Plot ─────────────────────────────────────────────────────────────────
    fig, ax = plt.subplots(figsize=(16, 13))

    # Words — small dots, coloured by dominant topic
    for k in range(K):
        mask = [i for i, t in enumerate(word_topic_labels) if t == k]
        if not mask:
            continue
        wc = word_coords[mask]
        ax.scatter(wc[:, 0], wc[:, 1], s=150, color=colors[k],
                   alpha=0.75, edgecolors='white', linewidths=0.5, zorder=2)

    # Word labels — only label words that are exclusively owned by one topic
    # (exclusivity > 0.5) to avoid a cluttered plot
    for i, (wx, wy) in enumerate(word_coords):
        word  = selected_words[i]
        dom_k = word_topic_labels[i]
        # exclusivity = fraction of this word's total beta belonging to dom_k
        excl  = beta[dom_k, selected_word_ids[i]] / (
                    beta[:, selected_word_ids[i]].sum() + 1e-10)
        if excl > 0.45:
            ax.annotate(word, (wx, wy), fontsize=6.5,
                        color=colors[dom_k], alpha=0.85,
                        xytext=(2, 2), textcoords="offset points")

    # Topic centroids — large stars with bold label
    for k, (tx, ty) in enumerate(topic_coords):
        top_word = topics_words[k][0]
        ax.scatter(tx, ty, s=350, color=colors[k],
                   marker="*", edgecolors="black", linewidths=0.8,
                   zorder=5, label=f"T{k}: {top_word}")
        ax.annotate(f"T{k}", (tx, ty), fontsize=11, fontweight="bold",
                    color="black",
                    xytext=(5, 5), textcoords="offset points", zorder=6)

    ax.set_title(
        f"Word & Topic Vector Space (t-SNE)"
        f"top-{topn_words} words/topic coloured by dominant topic · "
        f"★ = topic centroid",
        fontsize=12, fontweight="bold"
    )
    ax.set_xlabel("Dim 1")
    ax.set_ylabel("Dim 2")
    ax.legend(title="Topics", fontsize=8, markerscale=1.2,
              loc="upper right", framealpha=0.85)
    ax.grid(True, alpha=0.15)

    plt.tight_layout()
    plt.savefig(out_dir / "word_topic_space.png", dpi=160, bbox_inches="tight")
    plt.close()
    print(f"  saved word_topic_space.png  ({n_words} words, {K} topic stars)")


# =============================================================================
# MAIN
# =============================================================================

def main(run_dir: str, query_words: list, data_csv: str = None):
    run_path = Path(run_dir)
    art_dir  = run_path / "artifacts"
    out_dir  = run_path / "analysis"
    out_dir.mkdir(exist_ok=True)

    print(f"\n{'='*55}")
    print(f"  Analysing run: {run_path.name}")
    print(f"{'='*55}\n")

    # ── Load artifacts ───────────────────────────────────────────────────────
    doc_topic    = np.load(art_dir / "doc_topic.npy")
    topic_word   = np.load(art_dir / "topic_word_prob.npy")
    topic_vecs   = np.load(art_dir / "topic_vectors.npy") \
                   if (art_dir / "topic_vectors.npy").exists() else None
    topics_words = json.loads((art_dir / "topics_words.json").read_text())
    vocab        = json.loads((art_dir / "id2word.json").read_text())

    # Word vectors only saved by VAE models — skip gracefully for LDA/NMF
    word_vecs_path = art_dir / "word_vectors.npy"

    print("── 1. Training curves ──────────────────────────────────────────")
    plot_training_curves(run_path / "training_log.csv", out_dir)

    print("\n── 2. Topic vector analysis ────────────────────────────────────")
    if topic_vecs is not None:
        plot_topic_similarity(topic_vecs, topics_words, out_dir)
    else:
        print("  [skip] topic_vectors.npy not found")
    plot_topic_entropy(topic_word, topics_words, out_dir)

    print("\n── 3. Word vector analysis ─────────────────────────────────────")
    if word_vecs_path.exists():
        word_vecs = np.load(word_vecs_path)
        nearest_word_neighbors(word_vecs, vocab, query_words,
                               topn=10, out_dir=out_dir)
    else:
        print("  [skip] word_vectors.npy not found (not a VAE model)")
    plot_word_topic_affinity(topic_word, vocab, topics_words, out_dir)

    print("\n── 4. Document-topic analysis ──────────────────────────────────")
    plot_doc_topic_analysis(doc_topic, out_dir)

    print("\n── 5. t-SNE: original cluster vs predicted topic ───────────────")
    if data_csv:
        plot_tsne_from_csv(Path(data_csv), doc_topic, topics_words, out_dir)
    else:
        print("  [skip] --data_csv not provided")

    print("\n── 6. Word + topic vector space ────────────────────────────────")
    if word_vecs_path.exists():
        word_vecs = np.load(word_vecs_path)
        beta_mat  = np.load(art_dir / "topic_word_prob.npy")
        if topic_vecs is not None:
            plot_word_topic_space(word_vecs, topic_vecs, vocab,
                                  topics_words, beta_mat, out_dir,
                                  topn_words=15)
        else:
            print("  [skip] topic_vectors.npy not found")
    else:
        print("  [skip] word_vectors.npy not found")

    print(f"\n✓ All outputs saved to {out_dir}/\n")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--run_dir", required=True,
                        help="Path to results/<run_id>")
    parser.add_argument("--query_words", nargs="+",
                        default=["python", "java", "sql", "linux", "error"],
                        help="Words to find nearest neighbors for")
    parser.add_argument("--data_csv", default=None,
                        help="Path to CSV with columns: text,cluster,tsne_x,tsne_y")
    args = parser.parse_args()
    main(args.run_dir, args.query_words, args.data_csv)


