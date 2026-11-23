import argparse
import shutil
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from kitab.config import DEFAULT_DB_DIR, DOCS_JSONL
from kitab.retriever import build_index, get_embeddings, load_documents


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--docs", type=Path, default=DOCS_JSONL, help="books_docs.jsonl to embed")
    ap.add_argument("--db-dir", type=Path, default=DEFAULT_DB_DIR, help="where to write the Chroma index")
    ap.add_argument("--reset", action="store_true", help="delete the existing index first")
    args = ap.parse_args()

    if not args.docs.exists():
        sys.exit(f"{args.docs} not found. Run scripts/data_clean.py first.")
    if args.reset and args.db_dir.exists():
        shutil.rmtree(args.db_dir)

    docs = load_documents(args.docs)
    print(f"Embedding {len(docs):,} books into {args.db_dir} ...")
    store = build_index(docs, args.db_dir, get_embeddings(),
                        on_progress=lambda done, total: print(f"  {done:,}/{total:,}", end="\r"))
    print(f"\nDone. {store._collection.count():,} books indexed.")


if __name__ == "__main__":
    main()
