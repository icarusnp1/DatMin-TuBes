# Indonesian IR System (Full Build)

Features:
- Preprocessing: tokenization, stopword removal, Porter(ID) stemming (Snowball-style rules)
- Inverted index
- TF-IDF + cosine similarity (VSM)
- Query processing and document ranking
- Feature selection hook (see notes)
- Extractive summarization

## Quickstart
1) Put your .txt documents into `data/raw/`
2) Build artifacts:
   python -m src.app build --data-dir data/raw
3) Run a query:
   python -m src.app query --data-dir data/raw --query "contoh query"

## Testing
pip install -r requirements.txt
pytest -q

## Swapping the stemmer
Replace `src/stemmer_porter_id.py` only. Keep its public functions:
- stem_word(word: str) -> str
- stem_tokens(tokens: list[str]) -> list[str]
