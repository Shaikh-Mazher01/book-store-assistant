from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent

DOCS_JSONL = ROOT / "data" / "processed" / "books_docs.jsonl"
DEFAULT_DB_DIR = ROOT / "books_db"
COLLECTION = "books"

EMBED_MODEL = "sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2"
RERANKER_MODEL = "BAAI/bge-reranker-v2-m3"

FETCH_K = 30   # candidates pulled from each retriever
RRF_K = 60     # reciprocal-rank-fusion constant

LANGUAGES = {"english": "English", "hindi": "Hindi", "urdu": "Urdu", "marathi": "Marathi"}
SOURCES = ["rekhtabooks", "gyaanstore", "99bookstores", "kitabay",
           "crossword", "theindianbookstore", "oswaalbooks", "pustakcafe"]
