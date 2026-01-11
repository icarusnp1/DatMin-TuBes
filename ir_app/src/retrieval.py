"""TF-IDF + cosine retrieval on top of an inverted index."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List, Tuple, Iterable
import math

from .indexing import InvertedIndex

@dataclass
class TfidfModel:
    idf: Dict[str, float]

def _l2_norm(vec: Dict[str, float]) -> float:
    return math.sqrt(sum(v*v for v in vec.values()))

def build_doc_vectors(index: InvertedIndex, model: TfidfModel) -> Dict[str, Dict[str, float]]:
    """Build sparse TF-IDF vectors for each document (term->weight)."""
    doc_vecs: Dict[str, Dict[str, float]] = {doc_id: {} for doc_id in index.doc_len.keys()}

    for term, plist in index.postings.items():
        idf = model.idf.get(term, 0.0)
        for doc_id, tf in plist.items():
            # raw tf (could use log scaling)
            doc_vecs[doc_id][term] = float(tf) * idf

    # optional: cosine uses norms, compute elsewhere
    return doc_vecs

def build_query_vector(query_terms: List[str], model: TfidfModel) -> Dict[str, float]:
    tf: Dict[str, int] = {}
    for t in query_terms:
        tf[t] = tf.get(t, 0) + 1
    qv: Dict[str, float] = {}
    for term, f in tf.items():
        if term in model.idf:
            qv[term] = float(f) * model.idf[term]
    return qv

def retrieve(
    query_vec: Dict[str, float],
    index: InvertedIndex,
    doc_vecs: Dict[str, Dict[str, float]],
    *,
    top_k: int = 10,
) -> List[Tuple[str, float]]:
    """Cosine similarity. Candidates are union of postings for query terms."""
    if not query_vec:
        return []

    candidates = set()
    for term in query_vec.keys():
        plist = index.postings.get(term)
        if plist:
            candidates.update(plist.keys())

    q_norm = _l2_norm(query_vec)
    if q_norm == 0.0:
        return []

    # Precompute document norms lazily for candidates
    scores: Dict[str, float] = {}

    for doc_id in candidates:
        dv = doc_vecs.get(doc_id, {})
        # dot product only on query terms (sparse)
        dot = 0.0
        for term, q_w in query_vec.items():
            d_w = dv.get(term)
            if d_w is not None:
                dot += q_w * d_w
        if dot == 0.0:
            continue
        d_norm = _l2_norm(dv)
        if d_norm == 0.0:
            continue
        scores[doc_id] = dot / (q_norm * d_norm)

    ranked = sorted(scores.items(), key=lambda x: x[1], reverse=True)
    return ranked[:top_k]
