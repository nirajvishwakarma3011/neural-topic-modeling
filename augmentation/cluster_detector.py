import numpy as np
import pandas as pd
import hdbscan
from sklearn.metrics import pairwise_distances, silhouette_samples
from sklearn.ensemble import IsolationForest
from sklearn.preprocessing import StandardScaler
from sentence_transformers import SentenceTransformer
from dataclasses import dataclass, field
from typing import List, Dict, Optional
import logging

logger = logging.getLogger(__name__)


@dataclass
class ClusterProfile:
    cluster_id: int
    size: int
    tightness: float
    silhouette: float
    anomaly_score: float
    centroid: np.ndarray
    doc_indices: List[int]
    is_rare_candidate: bool = False


class RareClusterDetector:
    """
    Identifies rare-class candidate clusters from an unlabeled corpus.

    Rare class geometric signature:
        - Small cluster size (relative to corpus)
        - Low intra-cluster spread (tight)
        - High silhouette score (genuinely separated, not noise)
    """

    def __init__(
        self,
        embedding_model: str = "all-MiniLM-L6-v2",
        min_cluster_size: int = 5,
        min_samples: int = 3,
        silhouette_threshold: float = 0.25,
        contamination: float = 0.2,
        max_rare_size_frac: float = 0.05,  # rare cluster must be < this fraction of corpus
        device: str = "cuda",
    ):
        self.embedding_model_name = embedding_model
        self.min_cluster_size = min_cluster_size
        self.min_samples = min_samples
        self.silhouette_threshold = silhouette_threshold
        self.contamination = contamination
        self.max_rare_size_frac = max_rare_size_frac
        self.device = device

        self.embedder = SentenceTransformer(embedding_model, device=device)
        self.embeddings: Optional[np.ndarray] = None
        self.labels: Optional[np.ndarray] = None
        self.profiles: Dict[int, ClusterProfile] = {}

    def fit(self, documents: List[str], batch_size: int = 128) -> "RareClusterDetector":
        logger.info(f"Embedding {len(documents)} documents...")
        self.embeddings = self.embedder.encode(
            documents,
            batch_size=batch_size,
            show_progress_bar=True,
            normalize_embeddings=True,
        )

        logger.info("Running HDBSCAN clustering...")
        clusterer = hdbscan.HDBSCAN(
            min_cluster_size=self.min_cluster_size,
            min_samples=self.min_samples,
            metric="euclidean",
            cluster_selection_method="eom",
            prediction_data=True,
        )
        self.labels = clusterer.fit_predict(self.embeddings)

        n_clusters = len(set(self.labels)) - (1 if -1 in self.labels else 0)
        n_noise = (self.labels == -1).sum()
        logger.info(f"Found {n_clusters} clusters, {n_noise} noise points")

        self._compute_cluster_profiles()
        self._score_rare_candidates()
        return self

    def _compute_cluster_profiles(self):
        cluster_ids = [c for c in set(self.labels) if c != -1]

        if len(cluster_ids) < 2:
            logger.warning("Fewer than 2 clusters — silhouette undefined.")
            sil_scores = np.zeros(len(self.labels))
        else:
            sil_scores = silhouette_samples(self.embeddings, self.labels)

        for cid in cluster_ids:
            mask = self.labels == cid
            pts = self.embeddings[mask]
            indices = np.where(mask)[0].tolist()

            if len(pts) > 1:
                pdist = pairwise_distances(pts, metric="euclidean")
                np.fill_diagonal(pdist, np.nan)
                tightness = float(np.nanmean(pdist))
            else:
                tightness = 0.0

            centroid = pts.mean(axis=0)
            silhouette = float(sil_scores[mask].mean())

            self.profiles[cid] = ClusterProfile(
                cluster_id=cid,
                size=int(mask.sum()),
                tightness=tightness,
                silhouette=silhouette,
                anomaly_score=0.0,
                centroid=centroid,
                doc_indices=indices,
            )

    def _score_rare_candidates(self):
        if len(self.profiles) < 3:
            logger.warning("Too few clusters to run IsolationForest reliably.")
            return

        df = pd.DataFrame([
            {"cluster_id": cid, "size": p.size, "tightness": p.tightness}
            for cid, p in self.profiles.items()
        ])

        X = StandardScaler().fit_transform(df[["size", "tightness"]])

        iso = IsolationForest(
            contamination=self.contamination,
            random_state=42,
            n_estimators=200,
        )
        df["iso_label"] = iso.fit_predict(X)
        df["iso_score"] = iso.decision_function(X)

        max_rare_size = int(self.max_rare_size_frac * len(self.embeddings))

        for _, row in df.iterrows():
            cid = int(row["cluster_id"])
            self.profiles[cid].anomaly_score = float(row["iso_score"])
            is_rare = (
                row["iso_label"] == -1
                and self.profiles[cid].silhouette >= self.silhouette_threshold
                and self.profiles[cid].size <= max_rare_size   # hard size cap
            )
            self.profiles[cid].is_rare_candidate = is_rare
        n_rare = sum(p.is_rare_candidate for p in self.profiles.values())
        logger.info(f"Size cap: {max_rare_size} docs ({self.max_rare_size_frac*100:.1f}% of {len(self.embeddings)}). Rare candidates after cap: {n_rare}")

    def get_rare_doc_indices(self) -> List[int]:
        indices = []
        for p in self.profiles.values():
            if p.is_rare_candidate:
                indices.extend(p.doc_indices)
        return sorted(indices)

    def get_summary_df(self) -> pd.DataFrame:
        rows = []
        for p in self.profiles.values():
            rows.append({
                "cluster_id": p.cluster_id,
                "size": p.size,
                "tightness": round(p.tightness, 4),
                "silhouette": round(p.silhouette, 4),
                "anomaly_score": round(p.anomaly_score, 4),
                "is_rare_candidate": p.is_rare_candidate,
            })
        return pd.DataFrame(rows).sort_values("anomaly_score")

    def print_summary(self):
        df = self.get_summary_df()
        print("\n=== Cluster Summary ===")
        print(df.to_string(index=False))
        rare = df[df["is_rare_candidate"]]
        print(f"\nRare candidate clusters: {len(rare)}")
        print(f"Total rare candidate documents: {sum(rare['size'])}")
