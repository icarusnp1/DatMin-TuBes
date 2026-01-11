from src.preprocessing import preprocess
from src.stemmer_porter_id import stem_word

def test_stem_protect_si():
    assert stem_word("organisasi") == "organisasi"  # protect -si (no removal of -i)

def test_remove_particle_and_possessive():
    # makanlah -> makan, bukunya -> buku
    assert stem_word("makanlah") == "makan"
    assert stem_word("bukunya") == "buku"

def test_preprocess_pipeline():
    toks = preprocess("Ini adalah bukunya.")
    assert "ini" not in toks  # stopword removed
    assert "buku" in toks      # stemmed
