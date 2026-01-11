from __future__ import annotations
from typing import Iterable, List, Set
import re
from .stemmer_porter_id import stem_tokens

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
    sw = (DEFAULT_STOPWORDS | DOMAIN_STOPWORDS) if stopwords is None else stopwords
    return [t for t in tokens if t not in sw]

def preprocess(text: str) -> List[str]:
    toks = tokenize(text)
    toks = remove_stopwords(toks)
    toks = stem_tokens(toks)
    toks = [t for t in toks if t]
    return toks
