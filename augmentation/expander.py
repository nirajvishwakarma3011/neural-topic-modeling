import torch
import numpy as np
from transformers import AutoTokenizer, AutoModelForCausalLM
from typing import List, Optional
import logging

logger = logging.getLogger(__name__)


EXPANSION_PROMPT_TEMPLATE = """You are expanding a short text document for a topic modeling system.

The document belongs to a specific, rare topic cluster. The key vocabulary for this topic cluster is:
{centroid_vocab}

Original document:
{document}

Write an expanded version of this document (2-3x longer) that:
1. Preserves the original meaning and topic exactly
2. Uses more of the topic-specific vocabulary listed above
3. Adds relevant context, examples, or elaboration
4. Keeps the same domain and style

Expanded document:"""


class DocumentExpander:
    """
    Expands rare-class candidate documents using a local LLM.
    Uses centroid vocabulary to steer expansion toward cluster topic.
    """

    def __init__(
        self,
        model_name: str = "mistralai/Mistral-7B-Instruct-v0.3",
        device_map: str = "auto",
        max_new_tokens: int = 256,
        temperature: float = 0.7,
        top_p: float = 0.9,
    ):
        self.max_new_tokens = max_new_tokens
        self.temperature = temperature
        self.top_p = top_p

        logger.info(f"Loading {model_name}...")
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.model = AutoModelForCausalLM.from_pretrained(
            model_name,
            device_map=device_map,
            torch_dtype=torch.float16,
        )
        self.model.eval()

    def get_centroid_vocab(
        self,
        centroid: np.ndarray,
        vectorizer,
        top_k: int = 20,
    ) -> str:
        """
        Project centroid back to vocabulary via TF-IDF feature weights.

        centroid is in sentence-transformer embedding space; vectorizer is the
        TF-IDF/Count vectorizer fitted on the BoW corpus. We retrieve the
        top-k vocab terms by weight from the centroid activation.

        When centroid dim != vocab dim (always the case with sbert centroids),
        falls back to nearest-neighbor docs and extracts their top-TF-IDF terms.
        """
        feature_names = vectorizer.get_feature_names_out()
        vocab_size = len(feature_names)

        if centroid.shape[0] == vocab_size:
            top_indices = np.argsort(centroid)[-top_k:][::-1]
            return ", ".join(feature_names[top_indices])

        # centroid is in sbert space — use it as a query to retrieve nearest docs
        # then aggregate TF-IDF weights across those docs
        from sklearn.metrics.pairwise import cosine_similarity
        if hasattr(self, "_embeddings") and self._embeddings is not None:
            sims = cosine_similarity(centroid.reshape(1, -1), self._embeddings)[0]
            top_doc_indices = np.argsort(sims)[-20:]
            tfidf_matrix = vectorizer.transform(
                [self._documents[i] for i in top_doc_indices]
            )
            mean_weights = np.asarray(tfidf_matrix.mean(axis=0)).flatten()
            top_indices = np.argsort(mean_weights)[-top_k:][::-1]
            return ", ".join(feature_names[top_indices])

        # fallback: return empty string (prompt still works, just less steered)
        return ""

    def expand_document(self, document: str, centroid_vocab: str) -> str:
        prompt = EXPANSION_PROMPT_TEMPLATE.format(
            centroid_vocab=centroid_vocab,
            document=document,
        )

        inputs = self.tokenizer(
            prompt,
            return_tensors="pt",
            truncation=True,
            max_length=1024,
        ).to(self.model.device)

        with torch.no_grad():
            outputs = self.model.generate(
                **inputs,
                max_new_tokens=self.max_new_tokens,
                temperature=self.temperature,
                top_p=self.top_p,
                do_sample=True,
                pad_token_id=self.tokenizer.eos_token_id,
            )

        generated = outputs[0][inputs["input_ids"].shape[1]:]
        return self.tokenizer.decode(generated, skip_special_tokens=True).strip()

    def expand_batch(
        self,
        documents: List[str],
        centroids: List[np.ndarray],
        vectorizer,
        embeddings: Optional[np.ndarray] = None,
    ) -> List[str]:
        # stash for centroid vocab projection via nearest-neighbor fallback
        self._documents = documents
        self._embeddings = embeddings

        expanded = []
        for i, (doc, centroid) in enumerate(zip(documents, centroids)):
            if i % 10 == 0:
                logger.info(f"Expanding document {i}/{len(documents)}...")
            vocab = self.get_centroid_vocab(centroid, vectorizer)
            expanded.append(self.expand_document(doc, vocab))
        return expanded
