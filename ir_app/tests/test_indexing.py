from src.indexing import build_inverted_index

def test_build_inverted_index_basic():
    doc_tokens = {
        "d1": ["buku", "ajar", "ajar"],
        "d2": ["ajar", "kelas"],
    }
    idx = build_inverted_index(doc_tokens)
    assert idx.N == 2
    assert idx.df["ajar"] == 2
    assert idx.postings["ajar"]["d1"] == 2
    assert idx.postings["kelas"]["d2"] == 1
