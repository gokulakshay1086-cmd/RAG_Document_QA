"""
Wraps a sentence-transformers model to turn text into vectors.

Loaded lazily (and once, as a singleton) since loading the model is
the slowest part of startup.
"""
from functools import lru_cache
from typing import List

import numpy as np
from sentence_transformers import SentenceTransformer

from app.config import settings


class EmbeddingModel:
    def __init__(self, model_name: str):
        self.model_name = model_name
        self._model = SentenceTransformer(model_name)
        self.dimension = self._model.get_sentence_embedding_dimension()

    def encode(self, texts: List[str]) -> np.ndarray:
        """Returns an (N, dim) float32 numpy array of L2-normalized embeddings."""
        if not texts:
            return np.zeros((0, self.dimension), dtype="float32")
        embeddings = self._model.encode(
            texts,
            convert_to_numpy=True,
            normalize_embeddings=True,  # so cosine similarity == dot product
            show_progress_bar=False,
        )
        return embeddings.astype("float32")

    def encode_one(self, text: str) -> np.ndarray:
        return self.encode([text])[0]


@lru_cache(maxsize=1)
def get_embedding_model() -> EmbeddingModel:
    return EmbeddingModel(settings.embedding_model)
