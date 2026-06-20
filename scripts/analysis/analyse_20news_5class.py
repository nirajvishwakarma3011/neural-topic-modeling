"""
5-class subset analysis of 20news_10_filtered.csv:
  rec.motorcycles, rec.sport.baseball, rec.sport.hockey, sci.crypt, soc.religion.christian
Outputs:
  - console: size + class distribution
  - 20news_5class_tsne.png
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from sklearn.manifold import TSNE
from sklearn.feature_extraction.text import TfidfVectorizer
from sentence_transformers import SentenceTransformer
import warnings
warnings.filterwarnings("ignore")

DATA_PATH = "data/20news_10_filtered.csv"
TARGET_CLASSES = [
    "rec.motorcycles",
    "rec.sport.baseball",
    "rec.sport.hockey",
    "sci.crypt",
    "soc.religion.christian",
]
SBERT_MODEL = "all-MiniLM-L6-v2"
TSNE_PERPLEXITY = 40
TSNE_SEED = 42
OUTPUT_PNG = "20news_5class_tsne.png"
OUTPUT_CSV = "data/20news_5class.csv"

COLORS = ["#e6194b", "#3cb44b", "#4363d8", "#f58231", "#911eb4"]


def load_subset(path, classes):
    df = pd.read_csv(path)
    mask = df[classes].sum(axis=1) == 1
    df = df[mask].copy()
    df["label"] = df[classes].idxmax(axis=1)
    return df[["id", "text", "label"]].reset_index(drop=True)


def print_distribution(df):
    total = len(df)
    print(f"\n{'='*52}")
    print(f"  20news 5-class subset")
    print(f"{'='*52}")
    print(f"  Total docs : {total}")
    print(f"{'─'*52}")
    vc = df["label"].value_counts()
    for cls in TARGET_CLASSES:
        n = vc.get(cls, 0)
        pct = 100 * n / total
        bar = "█" * int(pct / 2)
        print(f"  {cls:<28} {n:>5}  {pct:5.1f}%  {bar}")
    print(f"{'='*52}\n")


def embed_sbert(texts, model_name):
    print(f"Embedding {len(texts)} docs with {model_name} ...")
    model = SentenceTransformer(model_name)
    embs = model.encode(texts, batch_size=128, show_progress_bar=True,
                        normalize_embeddings=True)
    return embs


def run_tsne(embs, perplexity, seed):
    print(f"Running t-SNE (perplexity={perplexity}) ...")
    tsne = TSNE(n_components=2, perplexity=perplexity, random_state=seed,
                max_iter=1000, metric="cosine", init="pca")
    return tsne.fit_transform(embs)


def plot_tsne(coords, labels, classes, colors, out_path):
    fig, ax = plt.subplots(figsize=(11, 8))
    color_map = dict(zip(classes, colors))
    for cls, col in zip(classes, colors):
        mask = labels == cls
        ax.scatter(coords[mask, 0], coords[mask, 1],
                   c=col, alpha=0.55, s=14, linewidths=0, label=cls)

    ax.set_title("t-SNE of 20news 5-class subset\n(SBERT all-MiniLM-L6-v2 embeddings)",
                 fontsize=13, pad=12)
    ax.set_xlabel("t-SNE dim 1")
    ax.set_ylabel("t-SNE dim 2")
    ax.legend(title="Class", fontsize=9, title_fontsize=10,
              markerscale=2, framealpha=0.85, loc="best")
    ax.grid(True, linewidth=0.3, alpha=0.4)
    plt.tight_layout()
    plt.savefig(out_path, dpi=150)
    print(f"Saved → {out_path}")
    plt.close()


def main():
    df = load_subset(DATA_PATH, TARGET_CLASSES)
    print_distribution(df)

    # save subset with one-hot columns + label column
    full = pd.read_csv(DATA_PATH)
    subset = full[full[TARGET_CLASSES].sum(axis=1) == 1].copy()
    subset["label"] = subset[TARGET_CLASSES].idxmax(axis=1)
    subset.to_csv(OUTPUT_CSV, index=False)
    print(f"Saved subset → {OUTPUT_CSV}  ({len(subset)} rows)")

    embs = embed_sbert(df["text"].tolist(), SBERT_MODEL)
    coords = run_tsne(embs, TSNE_PERPLEXITY, TSNE_SEED)
    plot_tsne(coords, df["label"].values, TARGET_CLASSES, COLORS, OUTPUT_PNG)


if __name__ == "__main__":
    main()
