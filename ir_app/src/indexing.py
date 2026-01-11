"""Inverted index builder and persistence."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List, Tuple
import json
import math

@dataclass
class InvertedIndex:
    postings: Dict[str, Dict[str, int]]  # term -> {doc_id: tf}
    df: Dict[str, int]                   # term -> document frequency
    doc_len: Dict[str, int]              # doc_id -> number of terms (post stopword+stemming)
    N: int                               # number of documents

    def to_jsonable(self) -> dict:
        return {
            "postings": self.postings,
            "df": self.df,
            "doc_len": self.doc_len,
            "N": self.N,
        }

    @staticmethod
    def from_jsonable(d: dict) -> "InvertedIndex":
        return InvertedIndex(
            postings={k: {dk: int(tv) for dk, tv in v.items()} for k, v in d["postings"].items()},
            df={k: int(v) for k, v in d["df"].items()},
            doc_len={k: int(v) for k, v in d["doc_len"].items()},
            N=int(d["N"]),
        )

def build_inverted_index(doc_tokens: Dict[str, List[str]]) -> InvertedIndex:
    postings: Dict[str, Dict[str, int]] = {}
    df: Dict[str, int] = {}
    doc_len: Dict[str, int] = {}

    for doc_id, toks in doc_tokens.items():
        doc_len[doc_id] = len(toks)
        tf_map: Dict[str, int] = {}
        for t in toks:
            tf_map[t] = tf_map.get(t, 0) + 1
        for term, tf in tf_map.items():
            postings.setdefault(term, {})[doc_id] = tf

    for term, plist in postings.items():
        df[term] = len(plist)

    return InvertedIndex(postings=postings, df=df, doc_len=doc_len, N=len(doc_tokens))

def save_index(index: InvertedIndex, path: str) -> None:
    with open(path, "w", encoding="utf-8") as f:
        json.dump(index.to_jsonable(), f, ensure_ascii=False)

def load_index(path: str) -> InvertedIndex:
    with open(path, "r", encoding="utf-8") as f:
        d = json.load(f)
    return InvertedIndex.from_jsonable(d)

def compute_idf(index: InvertedIndex, *, smooth: bool = True) -> Dict[str, float]:
    idf: Dict[str, float] = {}
    N = index.N
    for term, df in index.df.items():
        if smooth:
            idf[term] = math.log((N + 1) / (df + 1)) + 1.0
        else:
            idf[term] = math.log(N / df) if df else 0.0
    return idf
