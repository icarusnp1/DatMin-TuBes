from __future__ import annotations
from typing import Dict, List, Tuple
import math

from .indexing import InvertedIndex

SparseVec = Dict[str, float]

class TfidfModel:
    def __init__(self, *, idf: Dict[str, float]):
        self.idf = idf

    def tfidf(self, tf: int, term: str) -> float:
        # log-TF * IDF
        return (1.0 + math.log(tf)) * self.idf.get(term, 0.0)

def build_doc_vectors(index: InvertedIndex, model: TfidfModel) -> Dict[str, SparseVec]:
    docs: Dict[str, SparseVec] = {}
    # accumulate tf-idf per doc
    for term, postings in index.items():
        idf = model.idf.get(term, 0.0)
        if idf == 0.0:
            continue
        for doc_id, tf in postings.items():
            vec = docs.setdefault(doc_id, {})
            vec[term] = model.tfidf(tf, term)
    # normalize
    for doc_id, vec in docs.items():
        norm = math.sqrt(sum(v*v for v in vec.values())) or 1.0
        for t in list(vec.keys()):
            vec[t] /= norm
    return docs

def build_query_vector(q_terms: List[str], model: TfidfModel) -> SparseVec:
    tf: Dict[str, int] = {}
    for t in q_terms:
        if t in model.idf:  # enforce feature selection implicitly
            tf[t] = tf.get(t, 0) + 1
    vec: SparseVec = {}
    for t, c in tf.items():
        vec[t] = model.tfidf(c, t)
    norm = math.sqrt(sum(v*v for v in vec.values())) or 1.0
    for t in list(vec.keys()):
        vec[t] /= norm
    return vec

def cosine(q: SparseVec, d: SparseVec) -> float:
    # dot product over smaller vector
    if not q or not d:
        return 0.0
    if len(q) > len(d):
        q, d = d, q
    return sum(v * d.get(t, 0.0) for t, v in q.items())

def retrieve(q_vec: SparseVec, index: InvertedIndex, doc_vecs: Dict[str, SparseVec], *, top_k: int = 10) -> List[Tuple[str, float]]:
    # candidate docs from query terms postings
    cands = set()
    for t in q_vec.keys():
        postings = index.get(t)
        if postings:
            cands.update(postings.keys())
    scored = []
    for doc_id in cands:
        score = cosine(q_vec, doc_vecs.get(doc_id, {}))
        if score > 0:
            scored.append((doc_id, score))
    scored.sort(key=lambda x: x[1], reverse=True)
    return scored[:top_k]
