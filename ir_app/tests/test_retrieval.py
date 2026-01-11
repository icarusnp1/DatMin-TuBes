from src.indexing import build_inverted_index, compute_idf
from src.retrieval import TfidfModel, build_doc_vectors, build_query_vector, retrieve

def test_retrieval_prefers_more_matching_terms():
    doc_tokens = {
        "d1": ["buku", "ajar", "ajar"],
        "d2": ["buku"],
        "d3": ["kelas", "ajar"],
    }
    idx = build_inverted_index(doc_tokens)
    idf = compute_idf(idx, smooth=True)
    model = TfidfModel(idf=idf)
    doc_vecs = build_doc_vectors(idx, model)

    q_vec = build_query_vector(["buku", "ajar"], model)
    ranked = retrieve(q_vec, idx, doc_vecs, top_k=3)
    assert ranked[0][0] == "d1"  # has both terms, and higher tf for ajar than d3
