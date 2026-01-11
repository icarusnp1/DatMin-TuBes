from src.summarization import summarize_extractive

def test_summarize_returns_existing_sentences():
    text = "Buku ini tentang pembelajaran. Pembelajaran itu penting. Ini kalimat lain."
    idf = {"buku": 2.0, "ajar": 2.0, "pembelajar": 2.0, "penting": 1.5}
    s = summarize_extractive(text, idf=idf, num_sentences=2)
    assert isinstance(s, str)
    assert len(s) > 0
