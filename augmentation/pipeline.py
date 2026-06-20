import numpy as np
import logging
from typing import List, Optional
from .cluster_detector import RareClusterDetector
from .expander import DocumentExpander

logger = logging.getLogger(__name__)


class RareClassAugmentationPipeline:
    """
    Full pipeline:
        1. Detect rare-class candidates via geometric cluster analysis
        2. Expand candidates using LLM (or duplicate if no expander)
        3. Return augmented corpus for retraining
    """

    def __init__(
        self,
        detector: RareClusterDetector,
        expander: Optional[DocumentExpander] = None,
        augmentation_multiplier: int = 3,
    ):
        self.detector = detector
        self.expander = expander
        self.augmentation_multiplier = augmentation_multiplier

        self.rare_indices_: List[int] = []
        self.augmented_docs_: List[str] = []

    def fit_detect(self, documents: List[str]) -> "RareClassAugmentationPipeline":
        self.detector.fit(documents)
        self.rare_indices_ = self.detector.get_rare_doc_indices()
        n_rare_clusters = sum(p.is_rare_candidate for p in self.detector.profiles.values())
        logger.info(
            f"Detected {len(self.rare_indices_)} rare-candidate documents "
            f"across {n_rare_clusters} clusters"
        )
        return self

    def augment(self, documents: List[str], vectorizer=None) -> List[str]:
        """
        Returns augmented corpus:
        original documents + N expanded copies of each rare candidate.
        """
        if not self.rare_indices_:
            logger.warning("No rare candidates detected. Returning original corpus.")
            return documents

        rare_docs = [documents[i] for i in self.rare_indices_]

        rare_centroids = []
        for idx in self.rare_indices_:
            cluster_id = self.detector.labels[idx]
            rare_centroids.append(self.detector.profiles[cluster_id].centroid)

        if self.expander is None or vectorizer is None:
            logger.info(
                f"No expander/vectorizer — duplicating {len(rare_docs)} rare docs "
                f"x{self.augmentation_multiplier}."
            )
            augmented = list(documents)
            for _ in range(self.augmentation_multiplier):
                augmented.extend(rare_docs)
            self.augmented_docs_ = augmented[len(documents):]
            logger.info(f"Corpus: {len(documents)} → {len(augmented)} documents")
            return augmented

        logger.info(f"Expanding {len(rare_docs)} rare documents x{self.augmentation_multiplier}...")
        augmented = list(documents)

        for _ in range(self.augmentation_multiplier):
            expanded = self.expander.expand_batch(
                rare_docs,
                rare_centroids,
                vectorizer,
                embeddings=self.detector.embeddings,
            )
            augmented.extend(expanded)
            logger.info(f"  Added {len(expanded)} expanded documents.")

        self.augmented_docs_ = augmented[len(documents):]
        logger.info(f"Corpus: {len(documents)} → {len(augmented)} documents")
        return augmented

    def get_rare_cluster_summary(self):
        return self.detector.get_summary_df()
