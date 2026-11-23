"""Query understanding and filter logic."""
import re

from .config import LANGUAGES, SOURCES

# Query words that carry no signal for keyword (BM25) matching.
STOPWORDS = {
    "a", "an", "the", "on", "about", "of", "for", "from", "in", "to", "and", "or",
    "book", "books", "novel", "novels", "recommend", "suggest", "show", "me", "some",
    "under", "below", "above", "over", "less", "more", "than", "between", "rs", "inr",
    "rupees", "within", "upto", "up", "at", "least", "cheaper", "costlier", "good", "best",
}

NUM = r"(\d[\d,]*(?:\.\d+)?)"
CUR = r"(?:rs\.?|inr|₹|rupees)?\s*"
MAX_RE = re.compile(rf"(?:under|below|less than|cheaper than|within|upto|up to|not more than|<)\s*{CUR}{NUM}")
MIN_RE = re.compile(rf"(?:over|above|more than|at least|costlier than|>)\s*{CUR}{NUM}")
BETWEEN_RE = re.compile(rf"between\s*{CUR}{NUM}\s*(?:and|to|-)\s*{CUR}{NUM}")
TOKEN_RE = re.compile(r"[\w\u0900-\u097F\u0600-\u06FF]+")


def _num(s: str) -> float:
    return float(s.replace(",", ""))


def tokenize(text: str) -> list[str]:
    return TOKEN_RE.findall(text.lower())


def parse(query: str, sources: list[str] | None = None) -> tuple[dict, str]:
    sources = SOURCES if sources is None else sources
    q = query.lower()
    cleaned = q
    filters: dict = {}

    langs = [name for key, name in LANGUAGES.items() if re.search(rf"\b{key}\b", q)]
    if langs:  # "Hindi or Marathi" keeps both
        filters["languages"] = langs
        cleaned = re.sub(rf"\b(?:{'|'.join(LANGUAGES)})\b", " ", cleaned)

    if m := BETWEEN_RE.search(q):
        a, b = _num(m.group(1)), _num(m.group(2))
        filters["min_price"], filters["max_price"] = min(a, b), max(a, b)
        cleaned = BETWEEN_RE.sub(" ", cleaned)
    else:
        if m := MAX_RE.search(q):
            filters["max_price"] = _num(m.group(1))
            cleaned = MAX_RE.sub(" ", cleaned)
        if m := MIN_RE.search(q):
            filters["min_price"] = _num(m.group(1))
            cleaned = MIN_RE.sub(" ", cleaned)

    squashed = re.sub(r"[\s_-]", "", q)
    found = [s for s in sources if s in squashed]
    if found:
        filters["sources"] = found
        for s in found:
            cleaned = re.sub(r"[\s_-]*".join(map(re.escape, s)), " ", cleaned)

    cleaned = re.sub(r"\s+", " ", cleaned).strip()
    return filters, (cleaned if len(cleaned) >= 2 else query.strip())


def to_chroma(f: dict):
    clauses = []
    if f.get("languages"):
        langs = f["languages"]
        clauses.append({"language": {"$eq": langs[0]}} if len(langs) == 1
                       else {"language": {"$in": langs}})
    if f.get("sources"):
        srcs = f["sources"]
        clauses.append({"source": {"$eq": srcs[0]}} if len(srcs) == 1
                       else {"source": {"$in": srcs}})
    if "min_price" in f:
        clauses.append({"price": {"$gte": f["min_price"]}})
    if "max_price" in f:
        clauses.append({"price": {"$lte": f["max_price"]}})
    if not clauses:
        return None
    return clauses[0] if len(clauses) == 1 else {"$and": clauses}


def matches(meta: dict, f: dict) -> bool:
    if f.get("languages") and meta.get("language") not in f["languages"]:
        return False
    if f.get("sources") and meta.get("source") not in f["sources"]:
        return False
    price = meta.get("price")
    if "min_price" in f and (price is None or price < f["min_price"]):
        return False
    if "max_price" in f and (price is None or price > f["max_price"]):
        return False
    return True


def describe_filters(f: dict) -> str:
    parts = []
    if f.get("languages"):
        parts.append(" / ".join(x for x in f["languages"] if x != "Unknown") or "any language")
    if f.get("sources"):
        parts.append(", ".join(f["sources"]))
    lo, hi = f.get("min_price"), f.get("max_price")
    if lo is not None and hi is not None:
        parts.append(f"Rs {lo:g}-{hi:g}")
    elif hi is not None:
        parts.append(f"up to Rs {hi:g}")
    elif lo is not None:
        parts.append(f"Rs {lo:g} and above")
    return "  |  ".join(parts)
