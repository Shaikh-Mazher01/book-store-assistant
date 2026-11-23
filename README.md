# Book Store Assistant (Kitab)

A RAG (Retrieval-Augmented Generation) web app that helps readers find books across eight Indian online
bookstores. Ask in English, Hindi, Urdu or Marathi, optionally add a price or store ("Urdu poetry
between 100 and 500", "a love story from kitabay"), and get matching books with prices, links and a
short written recommendation.

**Live app:** https://bookassitant.streamlit.app/

> The catalog is a **snapshot** scraped from the stores' public product feeds, not a live feed.
> Prices and stock can change after the snapshot date shown in the app's sidebar. There are no ratings.

## How it works

```
question ──► parse filters (language, price, store) ──► hybrid retrieval ──► Groq LLM ──► answer
                                                          ├─ vector search (multilingual MiniLM, Chroma)
                                                          ├─ BM25 keyword search (filtered in Python)
                                                          └─ fused with reciprocal-rank fusion,
                                                             optional cross-encoder rerank
```

Each book is one document (title, author, language, format, topics, description). Price, store and
link live in metadata, so they can be filtered exactly and are never "remembered" by the model.

## Quick start

Requires Python 3.10+.

```bash
python -m venv .venv && source .venv/bin/activate      # Windows: .venv\Scripts\activate
pip install -r requirements.txt
cp .env.example .env                                   # add your GROQ_API_KEY
streamlit run app.py
```

On the first run the app downloads the embedding model (~470 MB) and builds the search index from
`data/processed/books_docs.jsonl` into `books_db/`. That takes a few minutes on CPU, once.
Without a `GROQ_API_KEY` the app still works, it just shows matching books without a written answer.

You can also build the index ahead of time: `python scripts/build_index.py` (add `--reset` to rebuild).

## Rebuilding the data

```bash
pip install -r requirements-dev.txt
python scripts/scrap.py          # data/raw/raw_data.csv        (scrape the stores)
python scripts/data_clean.py     # data/processed/books_docs.jsonl (+ clean_data.csv)
python scripts/build_index.py --reset
```

`data_clean.py` drops non-book items and implausible prices, merges the same book listed by several
stores (keeping the cheapest offer and listing the others), and writes one document per book.

## Project layout

```
app.py                  Streamlit UI
kitab/
  config.py             constants and paths
  query.py              turn "Hindi books under 300" into filters (pure Python)
  retriever.py          load documents, build the Chroma index, hybrid retriever
scripts/
  scrap.py              scrape the stores            -> data/raw/
  data_clean.py         clean + dedupe + make docs   -> data/processed/
  build_index.py        embed the docs               -> books_db/
data/processed/books_docs.jsonl   the cleaned catalog (committed)
tests/                  pytest suite (fake embedder; no model download needed)
```

## Configuration

| Variable | Purpose | Default |
|---|---|---|
| `GROQ_API_KEY` | enables written answers | unset (answers off) |
| `GROQ_MODEL` | Groq model used for answers | `qwen/qwen3.8-27b` |
| `BOOKS_DB_DIR` | where the Chroma index lives | `./books_db` |

## Tests

```bash
pip install -r requirements-dev.txt
pytest
```

## Deploying (e.g. Streamlit Community Cloud)

Add `GROQ_API_KEY` in the app's secrets. `books_db/` is git-ignored and rebuilt from the committed
`books_docs.jsonl` on first start. Cloud disks are not guaranteed to persist across restarts, so the
index may be rebuilt after one; if that is too slow, commit `books_db/` with Git LFS instead
(see the note in `.gitignore`) and check that your host fetches LFS files.

## Known data limits (Nov 2026 snapshot, 17,702 books)

- About 83% of books have no author in the store data, so author searches only work when the name
  appears in the title or description.
- About 27% have no language and are labelled "Unknown" (the sidebar can include or exclude them).
- The four largest stores were capped at 2,000 products per collection, so the catalog is partial.
- Exam-prep guides (mostly Oswaal) are included.
