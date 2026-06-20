# analysis.py
import argparse
import json
from pathlib import Path
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from scipy.spatial.distance import pdist, squareform
from scipy.stats import entropy
import warnings
from sklearn.manifold import TSNE

def plot_tsne_topics(topic_vectors, topics_words, out_dir, perplexity=10, random_state=42):
    """
    t-SNE visualization of topics using topic_vectors (K×D or K×V).
    """
    if topic_vectors is None:
        print("[warn] topic_vectors is None; skipping topic t-SNE.")
        return

    n_topics = topic_vectors.shape[0]
    if n_topics < 3:
        print("[warn] Too few topics for t-SNE; skipping.")
        return

    # Perplexity must be < n_samples
    perplexity = min(perplexity, max(2, n_topics - 1))

    tsne = TSNE(
        n_components=2,
        perplexity=perplexity,
        metric="cosine",
        random_state=random_state,
        init="random",
        learning_rate="auto",
    )
    emb = tsne.fit_transform(topic_vectors)

    plt.figure(figsize=(7, 6))
    plt.scatter(emb[:, 0], emb[:, 1])
    plt.title("t-SNE of Topics (cosine)")

    for i, (x, y) in enumerate(emb):
        snippet = ""
        if topics_words and i < len(topics_words):
            snippet = " " + " ".join(topics_words[i][:3])
        plt.annotate(f"{i}{snippet}", (x, y), fontsize=8)

    plt.tight_layout()
    plt.savefig(out_dir / "tsne_topics.png")
    plt.close()


def plot_tsne_documents(doc_topic, out_dir, perplexity=30, random_state=42, max_docs=5000):
    """
    t-SNE visualization of documents using doc-topic matrix (N×K).
    Colors by dominant topic.
    """
    if doc_topic is None or len(doc_topic) == 0:
        print("[warn] doc_topic missing/empty; skipping doc t-SNE.")
        return

    X = doc_topic
    n = X.shape[0]

    if n > max_docs:
        idx = np.random.RandomState(random_state).choice(n, size=max_docs, replace=False)
        X = X[idx]

    labels = np.argmax(X, axis=1)

    perplexity = min(perplexity, max(5, X.shape[0] // 3))

    tsne = TSNE(
        n_components=2,
        perplexity=perplexity,
        metric="cosine",
        random_state=random_state,
        init="random",
        learning_rate="auto",
    )
    emb = tsne.fit_transform(X)

    plt.figure(figsize=(7, 6))
    plt.scatter(emb[:, 0], emb[:, 1], c=labels, s=6)
    plt.title("t-SNE of Documents (colored by dominant topic)")
    plt.tight_layout()
    plt.savefig(out_dir / "tsne_documents.png")
    plt.close()


def _try_import_umap():
    try:
        import umap  # from umap-learn
        return umap
    except Exception:
        return None


def plot_umap_topics(topic_vectors, topics_words, out_dir, n_neighbors=10, min_dist=0.1, random_state=42):
    """
    UMAP scatter of topics. Uses topic_vectors (K×D or K×V).
    Labels each point with topic id and saves a PNG.
    """
    umap = _try_import_umap()
    if umap is None:
        print("[warn] umap-learn not installed; skipping topic UMAP. Install via: pip install umap-learn")
        return

    if topic_vectors is None:
        print("[warn] topic_vectors is None; skipping topic UMAP.")
        return

    reducer = umap.UMAP(
        n_neighbors=n_neighbors,
        min_dist=min_dist,
        metric="cosine",
        random_state=random_state,
    )
    emb = reducer.fit_transform(topic_vectors)

    plt.figure(figsize=(7, 6))
    plt.scatter(emb[:, 0], emb[:, 1])
    plt.title("UMAP of Topics (cosine)")

    # annotate with topic id and a short top-word snippet
    for i, (x, y) in enumerate(emb):
        snippet = ""
        if topics_words and i < len(topics_words) and len(topics_words[i]) > 0:
            snippet = " " + " ".join(topics_words[i][:3])
        plt.annotate(f"{i}{snippet}", (x, y), fontsize=8)

    plt.tight_layout()
    plt.savefig(out_dir / "umap_topics.png")
    plt.close()


def plot_umap_documents(doc_topic, out_dir, n_neighbors=15, min_dist=0.1, random_state=42, max_docs=5000):
    """
    UMAP scatter of documents using doc_topic matrix (N×K).
    Colors by dominant topic.
    """
    umap = _try_import_umap()
    if umap is None:
        print("[warn] umap-learn not installed; skipping doc UMAP. Install via: pip install umap-learn")
        return

    if doc_topic is None or len(doc_topic) == 0:
        print("[warn] doc_topic missing/empty; skipping doc UMAP.")
        return

    X = doc_topic
    n = X.shape[0]
    if n > max_docs:
        # subsample for speed / readability
        idx = np.random.RandomState(random_state).choice(n, size=max_docs, replace=False)
        X = X[idx]

    labels = np.argmax(X, axis=1)

    reducer = umap.UMAP(
        n_neighbors=n_neighbors,
        min_dist=min_dist,
        metric="cosine",
        random_state=random_state,
    )
    emb = reducer.fit_transform(X)

    plt.figure(figsize=(7, 6))
    plt.scatter(emb[:, 0], emb[:, 1], c=labels, s=6)
    plt.title("UMAP of Documents (colored by dominant topic)")
    plt.tight_layout()
    plt.savefig(out_dir / "umap_documents.png")
    plt.close()


def load_artifacts(run_dir: Path):
    art = run_dir / "artifacts"
    return {
        "doc_topic": np.load(art / "doc_topic.npy"),
        "topics_words": json.loads((art / "topics_words.json").read_text()),
        "topic_word_prob": np.load(art / "topic_word_prob.npy"),
        "id2word": json.loads((art / "id2word.json").read_text()),
        "topic_vectors": np.load(art / "topic_vectors.npy") if (art / "topic_vectors.npy").exists() else None,
    }


def plot_topic_word_heatmap(topic_word_prob, out_dir, topk=10):
    # Take top-k words per topic for visualization
    top_indices = np.argsort(-topic_word_prob, axis=1)[:, :topk]
    heatmap_data = np.take_along_axis(topic_word_prob, top_indices, axis=1)

    plt.figure(figsize=(topk, topic_word_prob.shape[0] * 0.4))
    sns.heatmap(heatmap_data, cmap="viridis")
    plt.xlabel("Top words")
    plt.ylabel("Topics")
    plt.title("Topic–Word Probability Heatmap")
    plt.tight_layout()
    plt.savefig(out_dir / "topic_word_heatmap.png")
    plt.close()


def compute_topic_entropy(topic_word_prob, out_dir):
    ent = entropy(topic_word_prob, axis=1)
    np.savetxt(out_dir / "topic_entropy.csv", ent, delimiter=",", header="entropy", comments="")
    return ent


def compute_topic_diversity(topics_words, out_dir):
    all_words = [w for t in topics_words for w in t]
    td = len(set(all_words)) / len(all_words)
    (out_dir / "topic_diversity.txt").write_text(f"Topic Diversity: {td:.4f}")
    return td


def plot_inter_topic_distance(topic_vectors, out_dir):
    if topic_vectors is None:
        return

    dist = squareform(pdist(topic_vectors, metric="cosine"))

    plt.figure(figsize=(6, 5))
    sns.heatmap(dist, cmap="magma", square=True)
    plt.title("Inter-topic Cosine Distance")
    plt.tight_layout()
    plt.savefig(out_dir / "inter_topic_distance.png")
    plt.close()


def plot_doc_topic_confidence(doc_topic, out_dir):
    confidence = np.max(doc_topic, axis=1)

    plt.figure(figsize=(6, 4))
    sns.histplot(confidence, bins=30, kde=True)
    plt.xlabel("Max topic probability per document")
    plt.ylabel("Count")
    plt.title("Document–Topic Confidence")
    plt.tight_layout()
    plt.savefig(out_dir / "doc_topic_confidence.png")
    plt.close()


def main(run_path: str):
    run_dir = Path(run_path)
    analysis_dir = run_dir / "analysis"
    analysis_dir.mkdir(exist_ok=True)

    artifacts = load_artifacts(run_dir)

    # 1. Topic-word distributions
    plot_topic_word_heatmap(artifacts["topic_word_prob"], analysis_dir)

    # # 2. Topic entropy
    # compute_topic_entropy(artifacts["topic_word_prob"], analysis_dir)

    # # 3. Topic diversity
    # compute_topic_diversity(artifacts["topics_words"], analysis_dir)

    # # 4. Inter-topic distance
    # plot_inter_topic_distance(artifacts["topic_vectors"], analysis_dir)

    # # 5. Document-topic confidence
    # plot_doc_topic_confidence(artifacts["doc_topic"], analysis_dir)

    # # 6. UMAP visualizations
    # plot_umap_topics(
    #     artifacts["topic_vectors"],
    #     artifacts["topics_words"],
    #     analysis_dir,
    #     n_neighbors=10,
    #     min_dist=0.1,
    #     random_state=42,
    # )

    # plot_umap_documents(
    #     artifacts["doc_topic"],
    #     analysis_dir,
    #     n_neighbors=15,
    #     min_dist=0.1,
    #     random_state=42,
    #     max_docs=5000,
    # )
    # # 7. t-SNE visualizations
    # plot_tsne_topics(
    #     artifacts["topic_vectors"],
    #     artifacts["topics_words"],
    #     analysis_dir,
    #     perplexity=10,
    #     random_state=42,
    # )

    # plot_tsne_documents(
    #     artifacts["doc_topic"],
    #     analysis_dir,
    #     perplexity=30,
    #     random_state=42,
    #     max_docs=5000,
    # )


    print(f"[analysis] saved outputs to {analysis_dir}")




if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--run_dir", required=True, help="Path to results/<run_id>")
    args = parser.parse_args()
    main(args.run_dir)
