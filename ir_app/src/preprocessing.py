from __future__ import annotations
from typing import List, Set, Dict, Any
import re

from .stemmer_porter_id import stem_tokens
from Sastrawi.StopWordRemover.StopWordRemoverFactory import StopWordRemoverFactory

stop_factory = StopWordRemoverFactory()
data = stop_factory.get_stop_words()

DEFAULT_STOPWORDS: Set[str] = {
    "yang","dan","di","ke","dari","pada","untuk","dengan","atau","sebagai","adalah","itu","ini","oleh","dalam",
    "pada","akan","dapat","tidak","lebih","juga","sudah","telah","karena","sehingga","agar","maka","bagi","saat",
    "hingga","antara","serta","misalnya","yaitu","yakni","tersebut","para","sebuah","dalam","dengan","tanpa"
}

DOMAIN_STOPWORDS: Set[str] = {
    "tanaman","obat","herbal","tradisional","khasiat","digunakan","penggunaan","berdasarkan",
    "ramuan","bahan","cara","pembuatan","penyakit","kesehatan","secara"
}

TOKEN_RE = re.compile(r"[a-zA-Z]+")

def tokenize(text: str) -> List[str]:
    return [t.lower() for t in TOKEN_RE.findall(text or "")]

def remove_stopwords(tokens: List[str], stopwords: Set[str] | None = None) -> List[str]:
    sw = (set(data) | set(DOMAIN_STOPWORDS) | set(DEFAULT_STOPWORDS)) if stopwords is None else stopwords
    return [t for t in tokens if t not in sw]

def preprocess(text: str) -> List[str]:
    toks = tokenize(text)
    toks = remove_stopwords(toks)
    toks = stem_tokens(toks)
    toks = [t for t in toks if t]
    return toks

def preprocess_breakdown(text: str, *, max_tokens: int = 80, max_chars: int = 500) -> Dict[str, Any]:
    """
    Untuk ditampilkan di UI:
    - case folding (lower)
    - tokenizing
    - stopword removal
    - stemming
    """
    original = (text or "")
    case_folded = original.lower()

    toks = tokenize(original)
    toks_no_stop = remove_stopwords(toks)
    toks_stem = stem_tokens(toks_no_stop)
    toks_final = [t for t in toks_stem if t]

    def _trunc_tokens(xs: List[str]) -> Dict[str, Any]:
        shown = xs[:max_tokens]
        return {
            "count": len(xs),
            "shown": shown,
            "truncated": len(xs) > len(shown),
        }

    return {
        "original_preview": original[:max_chars] + ("..." if len(original) > max_chars else ""),
        "case_folding_preview": case_folded[:max_chars] + ("..." if len(case_folded) > max_chars else ""),
        "tokenize": _trunc_tokens(toks),
        "stopword_removed": _trunc_tokens(toks_no_stop),
        "stemming": _trunc_tokens(toks_final),
    }