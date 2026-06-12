#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Shared semantic-distance helpers for prompt robustness experiments.

The preferred backend uses sentence-transformers embeddings and cosine distance.
A deterministic TF-IDF word/character n-gram fallback is provided so experiments
can still run in environments where sentence-transformers or the configured
embedding model is unavailable.
"""

from __future__ import annotations

import importlib.util
import math
import re
import sys
from collections import Counter
from functools import lru_cache
from typing import Any, Iterable, Literal, Mapping, Sequence

DEFAULT_SENTENCE_TRANSFORMER_MODEL = "sentence-transformers/all-MiniLM-L6-v2"
Backend = Literal["auto", "sentence-transformers", "tfidf"]

TOKEN_RE = re.compile(r"[a-z0-9]+(?:[-'][a-z0-9]+)?")


def cosine_distance(left: Iterable[float], right: Iterable[float]) -> float:
    """Return cosine distance, clamped to the interval [0, 1]."""
    left_values = list(left)
    right_values = list(right)
    dot = sum(a * b for a, b in zip(left_values, right_values))
    left_norm = math.sqrt(sum(a * a for a in left_values))
    right_norm = math.sqrt(sum(b * b for b in right_values))
    if left_norm == 0 or right_norm == 0:
        return 1.0
    similarity = dot / (left_norm * right_norm)
    return float(max(0.0, min(1.0, 1.0 - similarity)))


@lru_cache(maxsize=4)
def _load_sentence_transformer(model_name: str) -> Any:
    """Load and cache a sentence-transformers model by name/path."""
    from sentence_transformers import SentenceTransformer  # type: ignore[import-not-found]

    return SentenceTransformer(model_name)


def sentence_transformer_pair_distances(
    text_pairs: Sequence[tuple[str, str]],
    *,
    model_name: str = DEFAULT_SENTENCE_TRANSFORMER_MODEL,
) -> list[float]:
    """Compute cosine distances for text pairs with sentence-transformers."""
    model = _load_sentence_transformer(model_name)
    texts: list[str] = []
    for left, right in text_pairs:
        texts.extend([left, right])

    embeddings = model.encode(texts, normalize_embeddings=True, show_progress_bar=False)
    distances: list[float] = []
    for index in range(len(text_pairs)):
        distances.append(cosine_distance(embeddings[index * 2], embeddings[index * 2 + 1]))
    return distances


def sentence_transformer_reference_distances(
    query: str,
    references: Mapping[str, str],
    *,
    model_name: str = DEFAULT_SENTENCE_TRANSFORMER_MODEL,
) -> dict[str, float]:
    """Compute query-to-reference cosine distances with sentence-transformers."""
    model = _load_sentence_transformer(model_name)
    reference_items = list(references.items())
    texts = [query, *[reference_text for _, reference_text in reference_items]]
    embeddings = model.encode(texts, normalize_embeddings=True, show_progress_bar=False)
    query_embedding = embeddings[0]
    return {
        reference_key: cosine_distance(query_embedding, embeddings[index + 1])
        for index, (reference_key, _) in enumerate(reference_items)
    }


def tokenise(text: str) -> list[str]:
    """Tokenise text for the TF-IDF backend."""
    return TOKEN_RE.findall(text.lower())


def ngrams(items: Sequence[str], minimum: int, maximum: int) -> list[str]:
    """Return inclusive n-grams from a token or character sequence."""
    output: list[str] = []
    for n in range(minimum, maximum + 1):
        output.extend(" ".join(items[index : index + n]) for index in range(max(0, len(items) - n + 1)))
    return output


def tfidf_features(text: str) -> Counter[str]:
    """Build word and character n-gram counts for the TF-IDF backend."""
    tokens = tokenise(text)
    compact_chars = list(re.sub(r"\s+", " ", text.lower()).strip())
    features: Counter[str] = Counter()
    features.update(f"word:{gram}" for gram in ngrams(tokens, 1, 2))
    features.update(f"char:{gram}" for gram in ngrams(compact_chars, 3, 5))
    return features


def _tfidf_vectors(documents: Sequence[str]) -> list[dict[str, float]]:
    counts = [tfidf_features(text) for text in documents]
    document_count = len(counts)
    document_frequency: Counter[str] = Counter()
    for features in counts:
        document_frequency.update(features.keys())

    return [
        {
            feature: term_frequency * (math.log((1 + document_count) / (1 + document_frequency[feature])) + 1.0)
            for feature, term_frequency in features.items()
        }
        for features in counts
    ]


def sparse_cosine_distance(left: Mapping[str, float], right: Mapping[str, float]) -> float:
    """Return cosine distance for sparse vectors, clamped to [0, 1]."""
    if len(left) > len(right):
        left, right = right, left
    dot = sum(value * right.get(feature, 0.0) for feature, value in left.items())
    left_norm = math.sqrt(sum(value * value for value in left.values()))
    right_norm = math.sqrt(sum(value * value for value in right.values()))
    if left_norm == 0 or right_norm == 0:
        return 1.0
    similarity = dot / (left_norm * right_norm)
    return float(max(0.0, min(1.0, 1.0 - similarity)))


def tfidf_pair_distances(text_pairs: Sequence[tuple[str, str]]) -> list[float]:
    """Compute cosine distances for text pairs with TF-IDF n-gram vectors."""
    documents = [text for text_pair in text_pairs for text in text_pair]
    vectors = _tfidf_vectors(documents)
    distances: list[float] = []
    for index in range(len(text_pairs)):
        distances.append(sparse_cosine_distance(vectors[index * 2], vectors[index * 2 + 1]))
    return distances


def tfidf_reference_distances(query: str, references: Mapping[str, str]) -> dict[str, float]:
    """Compute query-to-reference cosine distances with TF-IDF n-gram vectors."""
    reference_items = list(references.items())
    vectors = _tfidf_vectors([query, *[reference_text for _, reference_text in reference_items]])
    query_vector = vectors[0]
    return {
        reference_key: sparse_cosine_distance(query_vector, vectors[index + 1])
        for index, (reference_key, _) in enumerate(reference_items)
    }


def _sentence_transformers_available() -> bool:
    return importlib.util.find_spec("sentence_transformers") is not None


def _fallback_message(reason: str, *, warn: bool) -> None:
    if warn:
        print(f"⚠️ Falling back to TF-IDF semantic-distance proxy: {reason}", file=sys.stderr)


def pair_distances(
    text_pairs: Sequence[tuple[str, str]],
    *,
    backend: Backend = "auto",
    model_name: str = DEFAULT_SENTENCE_TRANSFORMER_MODEL,
    warn_on_fallback: bool = True,
) -> tuple[list[float], str]:
    """Compute text-pair cosine distances and return the backend actually used."""
    if backend in {"auto", "sentence-transformers"}:
        if not _sentence_transformers_available():
            if backend == "sentence-transformers":
                raise RuntimeError("sentence-transformers is not installed. Install it or use backend='tfidf'.")
            _fallback_message("sentence-transformers is not installed", warn=warn_on_fallback)
        else:
            try:
                return sentence_transformer_pair_distances(text_pairs, model_name=model_name), "sentence-transformers"
            except Exception as exc:
                if backend == "sentence-transformers":
                    raise RuntimeError(
                        "Could not compute sentence-transformers distances. "
                        "Ensure the embedding model is available, or use backend='tfidf'."
                    ) from exc
                _fallback_message(str(exc), warn=warn_on_fallback)

    return tfidf_pair_distances(text_pairs), "tfidf"


def reference_distances(
    query: str,
    references: Mapping[str, str],
    *,
    backend: Backend = "auto",
    model_name: str = DEFAULT_SENTENCE_TRANSFORMER_MODEL,
    warn_on_fallback: bool = True,
) -> tuple[dict[str, float], str]:
    """Compute query-to-reference cosine distances and return the backend actually used."""
    if backend in {"auto", "sentence-transformers"}:
        if not _sentence_transformers_available():
            if backend == "sentence-transformers":
                raise RuntimeError("sentence-transformers is not installed. Install it or use backend='tfidf'.")
            _fallback_message("sentence-transformers is not installed", warn=warn_on_fallback)
        else:
            try:
                return (
                    sentence_transformer_reference_distances(query, references, model_name=model_name),
                    "sentence-transformers",
                )
            except Exception as exc:
                if backend == "sentence-transformers":
                    raise RuntimeError(
                        "Could not compute sentence-transformers distances. "
                        "Ensure the embedding model is available, or use backend='tfidf'."
                    ) from exc
                _fallback_message(str(exc), warn=warn_on_fallback)

    return tfidf_reference_distances(query, references), "tfidf"
