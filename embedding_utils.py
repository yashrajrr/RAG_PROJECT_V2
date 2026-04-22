from __future__ import annotations

import hashlib
import re

import numpy as np

EMBEDDING_DIMENSION = 384
TOKEN_PATTERN = re.compile(r"\b\w+\b")


def _tokenize(text: str) -> list[str]:
    return TOKEN_PATTERN.findall(text.lower())


def _token_to_index(token: str) -> int:
    digest = hashlib.blake2b(token.encode("utf-8"), digest_size=8).digest()
    return int.from_bytes(digest, "little") % EMBEDDING_DIMENSION


def create_embeddings(texts: list[str], batch_size: int = 64) -> np.ndarray:
    if not texts:
        return np.empty((0, EMBEDDING_DIMENSION), dtype=np.float32)

    embeddings: list[np.ndarray] = []
    for text in texts:
        vector = np.zeros(EMBEDDING_DIMENSION, dtype=np.float32)
        tokens = _tokenize(text)
        if not tokens:
            embeddings.append(vector)
            continue

        for token in tokens:
            vector[_token_to_index(token)] += 1.0

        norm = np.linalg.norm(vector)
        if norm > 0:
            vector /= norm
        embeddings.append(vector)

    return np.vstack(embeddings)


def cosine_similarity(matrix: np.ndarray, vector: np.ndarray) -> np.ndarray:
    if matrix.ndim != 2:
        raise ValueError("Expected a 2D matrix of embeddings.")
    if vector.ndim != 1:
        raise ValueError("Expected a 1D query embedding.")

    matrix = np.asarray(matrix, dtype=np.float32)
    vector = np.asarray(vector, dtype=np.float32)

    matrix_norms = np.linalg.norm(matrix, axis=1)
    vector_norm = np.linalg.norm(vector)

    safe_denominator = np.clip(matrix_norms * max(vector_norm, 1e-12), 1e-12, None)
    return (matrix @ vector) / safe_denominator
