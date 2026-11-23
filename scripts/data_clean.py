import datetime
import json
import os
import re
import string
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
RAW_CSV = ROOT / "data" / "raw" / "raw_data.csv"
CLEAN_CSV = ROOT / "data" / "processed" / "clean_data.csv"
DOCS_JSONL = ROOT / "data" / "processed" / "books_docs.jsonl"

MIN_PRICE, MAX_PRICE = 20, 5000
MAX_DESC_CHARS = 1200
MAX_TAGS = 8

BAD_CHARS = re.compile(r"[\ufffd\u200b\u200e\u200f\ufeff]")
PUNCT = re.compile("[" + re.escape(string.punctuation) + "।॥]")
TAG_NOISE = re.compile(
    r"^[A-Za-z]+:\S*$|^\d+-[A-Z]\d+$|^(?:books\d*|new book|all books|gently used|nsp|fiction)$", re.I)
TAG_JUNK = re.compile(r"^(?:19|20)\d{2}$|^\d+(?:st|nd|rd|th)$|^\d+$", re.I)
SET_RE = re.compile(r"box ?set|\bset of\b|\bcollection\b|\b\d+ books\b|complete (?:set|series)", re.I)


HAS_LETTERS = re.compile(r"[A-Za-z\u0900-\u097F\u0600-\u06FF]{3}")

FAKE_DESC = {"na", "n/a", "none", "null", "nil", "-", "--", "."}


def clean(s):
    return re.sub(r"\s+", " ", BAD_CHARS.sub(" ", s)).strip()


def title_key(title):
    t = re.sub(r"\(.*?\)|\[.*?\]", " ", title.lower())
    return re.sub(r"\s+", " ", PUNCT.sub(" ", t)).strip()


def build(raw_csv=RAW_CSV, clean_csv=CLEAN_CSV, docs_jsonl=DOCS_JSONL, scraped_at=None):
    """Run the whole cleaning step. Returns (summary DataFrame, list of documents)."""
    raw_csv = Path(raw_csv)
    if scraped_at is None:  # the scrape date is when the raw file was last written
        scraped_at = datetime.date.fromtimestamp(os.path.getmtime(raw_csv)).isoformat()

    df = pd.read_csv(raw_csv, dtype=str, keep_default_na=False)
    print(f"Raw rows: {len(df)}")
    for col in df.columns:
        df[col] = df[col].map(clean)

    # ---- filters: drop only rows with no title or an implausible price ----
    df["price"] = pd.to_numeric(df["price"], errors="coerce")
    df = df[df["price"].between(MIN_PRICE, MAX_PRICE) & (df["title"] != "")]
    print(f"After price filter: {len(df)}")

    # Fake / garbage descriptions become empty. The book itself is kept.
    no_letters = ~df["description"].map(lambda d: bool(HAS_LETTERS.search(d)))
    df.loc[df["description"].str.lower().isin(FAKE_DESC) | no_letters, "description"] = ""

    df["is_set"] = df["title"].map(lambda t: bool(SET_RE.search(t)))
    df["key"] = [f"{title_key(t)}|{int(s)}" for t, s in zip(df["title"], df["is_set"])]

    # Same title but a different language / author means a different book: split the key.
    for col in ("language", "author"):
        n = df[df[col] != ""].groupby("key")[col].nunique()
        conflict = df["key"].map(n).fillna(0) > 1
        df.loc[conflict, "key"] += "|" + df.loc[conflict, col].str.lower()

    df = df.sort_values("price", kind="stable")  # cheapest offer first within each group
    rows, docs = [], []
    for i, (key, g) in enumerate(df.groupby("key", sort=False)):
        best = g.iloc[0]
        author = next((a for a in g["author"] if a), "")
        language = next((l for l in g["language"] if l), "")
        fmt = next((f for f in g["format"] if f), "")
        description = max(g["description"], key=len)[:MAX_DESC_CHARS].strip()

        title_l = best["title"].lower()
        tags, seen = [], set()
        for t in " | ".join(g["tags"]).split(" | "):
            t = t.strip()
            if (t and not TAG_NOISE.match(t) and not TAG_JUNK.match(t)
                    and t.lower() != title_l and t.lower() not in seen):
                seen.add(t.lower())
                tags.append(t)
        tags = tags[:MAX_TAGS]

        cheapest_per_site = {}
        for s, p in zip(g["source"], g["price"]):
            cheapest_per_site.setdefault(s, p)
        offers = [f"{s}: Rs {p:g}" for s, p in cheapest_per_site.items()]
        num_sources = int(g["source"].nunique())

        rows.append({"title": best["title"], "author": author, "language": language, "format": fmt,
                     "price": best["price"], "cheapest_source": best["source"],
                     "all_prices": " | ".join(offers), "num_sources": num_sources,
                     "tags": " | ".join(tags), "book_url": best["book_url"],
                     "image_url": best["image_url"], "description": description,
                     "scraped_at": scraped_at})


        lines = [f"Title: {best['title']}"]
        if author:
            lines.append(f"Author: {author}")
        if language:
            lines.append(f"Language: {language}")
        if fmt:
            lines.append(f"Format: {fmt}")
        if tags:
            lines.append(f"Topics: {', '.join(tags)}")
        if description:
            lines.append(f"About: {description}")

        doc_id = f"book-{i}"
        docs.append({
            "id": doc_id,
            "text": "\n".join(lines),
            "metadata": {
                "id": doc_id,
                "title": best["title"],
                "author": author,
                "language": language or "Unknown",
                "format": fmt,
                "price": float(best["price"]),
                "source": best["source"],
                "all_prices": " | ".join(offers),
                "num_sources": num_sources,
                "url": best["book_url"],
                "image_url": best["image_url"],
                "scraped_at": scraped_at,
            },
        })

    out = pd.DataFrame(rows)
    Path(clean_csv).parent.mkdir(parents=True, exist_ok=True)
    Path(docs_jsonl).parent.mkdir(parents=True, exist_ok=True)
    out.to_csv(clean_csv, index=False, encoding="utf-8-sig")
    with open(docs_jsonl, "w", encoding="utf-8", newline="\n") as f:
        for d in docs:
            f.write(json.dumps(d, ensure_ascii=False) + "\n")
    return out, docs


def main():
    out, docs = build()
    print(f"\nUnique books: {len(out)} -> {CLEAN_CSV}, {DOCS_JSONL}")
    print(f"available on 2+ sites: {(out['num_sources'] > 1).sum()} | "
          f"author filled: {(out['author'] != '').mean():.0%} | "
          f"language filled: {(out['language'] != '').mean():.0%} | "
          f"description filled: {(out['description'] != '').mean():.0%}")
    print(f"Price range: Rs {out['price'].min():g} - Rs {out['price'].max():g}")
    lens = sorted(len(d["text"]) for d in docs)
    print(f"Doc length (characters): median {lens[len(lens)//2]}, max {lens[-1]}")


if __name__ == "__main__":
    main()
