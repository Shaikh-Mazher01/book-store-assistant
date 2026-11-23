import csv
import html
import re
import time
import warnings
from itertools import count
from pathlib import Path
from urllib.parse import urlparse

import pandas as pd
import requests
from bs4 import BeautifulSoup
from requests.adapters import HTTPAdapter
from urllib3.util.retry import Retry

try:
    from bs4 import XMLParsedAsHTMLWarning
    warnings.filterwarnings("ignore", category=XMLParsedAsHTMLWarning)
except ImportError:
    pass

ROOT = Path(__file__).resolve().parents[1]
OUTPUT_CSV = ROOT / "data" / "raw" / "raw_data.csv"
DELAY = 1.0
MAX_BOOKS_PER_SITE = 2000   # set to None to scrape everything from each website (can take more time)

Indian_bookstore_collections = ["fantasy", "game-books", "finance", "fiction","flip-flap-lift-and-look-books",
                                "hardcover-books", "historical-book", "motivation-book", "mystery-box", "non-fiction",
                                "phonic-books-early-reading", "rhyme-books", "romance-relationships"]

URLS = ["https://99bookstores.com",
        "https://pustakcafe.com",
        "https://kitabay.com",
        "https://gyaanstore.com",
        "https://www.crossword.in",
        "https://www.oswaalbooks.com"] +[
        f"https://www.theindianbookstore.in/collections/{c}" for c in Indian_bookstore_collections]+ [
        f"https://rekhtabooks.com/collections/{c}" for c in ["urdu-books", "hindi-books", "english-books", "marathi"]]

COLUMNS = ["source", "title", "author", "price", "vendor", "language", "format",
           "tags", "book_url", "image_url", "description"]
COLLECTION_LANG = {"urdu-books": "Urdu", "hindi-books": "Hindi",
                   "english-books": "English", "marathi": "Marathi"}

# Non-book merchandise that Shopify book stores also list. Matched against the title.
JUNK_TITLE = re.compile(
    r"key ?chain|key ?ring|die[- ]?cast|katana|ninjato|lightsaber|bath bombs?|"
    r"levitation|\(\d+ ?ounce\)|anime figure|action figure|bobble ?head|"
    r"fridge magnet|lenticular|rubber band game|chess game|bookmark|"
    r"\bpen\b|\bgsm\b|\bcombo\b|"
    r"projector|tapestry|wall hanging|scroll poster|wall (?:art|decor)|music box|"
    r"zentai|costume|charging station|hologram|building blocks|"
    r"\b(?:ruled|unruled) notebook\b|\bloope\b|\buno (?:playing )?cards?\b|playing cards?|"
    r"fan dual blade|\bmugs?\b",
    re.I,
)

# Tags that carry no information (ids, sale labels, sort keys, etc).
NOISE = re.compile(
    r"\d{5,}|\bsale\b|\babove\b|booksdesc|desclive|delivery|bestsell|bvmeta|"
    r"^author:|^bst$|^books$",re.I,
)

LANG_MAP = {"english": "English", "hindi": "Hindi", "urdu": "Urdu", "marathi": "Marathi"}
FORMAT_MAP = {"paperback": "Paperback", "paper back": "Paperback", "pb": "Paperback",
              "hardcover": "Hardcover", "hard cover": "Hardcover",
              "hardback": "Hardcover", "hb": "Hardcover"}


def is_junk(title):
    return bool(JUNK_TITLE.search(title or ""))


def make_session():
    s = requests.Session()
    s.headers.update({"User-Agent": "Mozilla/5.0 (personal project)"})
    retry = Retry(
        total=2,
        backoff_factor=1.5,
        status_forcelist=(429, 500, 502, 503, 504),
        allowed_methods=("GET",),
        respect_retry_after_header=True,
    )
    adapter = HTTPAdapter(max_retries=retry)
    s.mount("https://", adapter)
    s.mount("http://", adapter)
    return s


session = make_session()

def get(url):
    time.sleep(DELAY)
    r = session.get(url, timeout=25)
    r.raise_for_status()
    return r

# Helpers
def site_name(url):
    return urlparse(url).netloc.replace("www.", "").split(".")[0]

def site_base(url):
    p = urlparse(url)
    return f"{p.scheme}://{p.netloc}"

def clean_text(value):
    if value is None:
        return ""
    return re.sub(r"\s+", " ", html.unescape(str(value))).strip()

def html_to_text(raw_html):
    return BeautifulSoup(raw_html or "", "html.parser").get_text(" ")

def to_price(value):
    try:
        return float(value)
    except (TypeError, ValueError):
        return ""


ROLE = r"(?:written|illustrated|translated|adapted|edited|retold|compiled|narrated|foreword)"
ROLE_BY = re.compile(rf"\b{ROLE}\s+by\s+", re.I)
BY = re.compile(r"\s+by\s+", re.I)
ROLE_WORD_END = re.compile(rf"\b{ROLE}$", re.I)

def split_author(title):
    title = title or ""
    cuts = [m for m in BY.finditer(title) if not ROLE_WORD_END.search(title[:m.start()])]
    if not cuts:
        return title, ""
    m = cuts[-1]
    head, tail = title[:m.start()].strip(" -–:"), title[m.end():]
    tail = re.split(r"\s*[\(\[|]", tail)[0]
    tail = ROLE_BY.sub(", ", tail).strip(" ,-–")
    tail = re.sub(r"\s*,(?:\s*,)*\s*", ", ", tail).strip(" ,")
    n_head, n_auth = len(head.split()), len(tail.split())
    if head and tail and (n_head >= 2 or n_auth >= 2) and n_auth <= 8:
        return head, tail
    return title, ""


def clean_tags(tags):
    if isinstance(tags, str):
        tags = tags.split(",")
    tags = [str(t).strip() for t in (tags or []) if str(t).strip()]

    author, kept = "", []
    for t in tags:
        if t.lower().startswith("author:"):
            author = t.split(":", 1)[1].strip()
        elif not NOISE.search(t):
            kept.append(t)

    lang = next((LANG_MAP[t.lower()] for t in kept if t.lower() in LANG_MAP), "")
    fmt = next((FORMAT_MAP[t.lower()] for t in kept if t.lower() in FORMAT_MAP), "")
    rest = [t for t in kept if t.lower() not in LANG_MAP and t.lower() not in FORMAT_MAP]
    return author, lang, fmt, " | ".join(rest)

def guess_language(title):
    if re.search(r"[\u0600-\u06FF]", title or ""):
        return "Urdu"
    if re.search(r"[\u0900-\u097F]", title or ""):
        return "Hindi"
    return ""

def pick_variant(variants):
    variants = variants or [{}]
    good = [v for v in variants if (to_price(v.get("price")) or 0) > 0]
    in_stock = [v for v in good if v.get("available") is not False]
    return (in_stock or good or variants)[0]

def make_row(**fields):
    return {col: clean_text(fields.get(col)) for col in COLUMNS}


class BookScraper:
    def __init__(self, urls, output_csv=OUTPUT_CSV, max_per_site=MAX_BOOKS_PER_SITE):
        self.urls = urls
        self.output_csv = output_csv
        self.max_per_site = max_per_site
        self.seen = set()
        self.stats = {"kept": 0, "junk": 0, "duplicate": 0, "out_of_stock": 0}

    def shopify(self, url):
        collection = url.split("/collections/")[1].strip("/") if "/collections/" in url else ""
        fallback_lang = COLLECTION_LANG.get(collection, "")
        prev_first = None
        for page in count(1):
            data = get(f"{url.rstrip('/')}/products.json?limit=250&page={page}").json()
            products = data.get("products") or []
            if not products:
                return
            first_id = products[0].get("id")
            if first_id is not None and first_id == prev_first:
                return
            prev_first = first_id

            for p in products:
                variant = pick_variant(p.get("variants"))
                if variant.get("available") is False:
                    self.stats["out_of_stock"] += 1
                    continue

                title, title_author = split_author(clean_text(p.get("title")))
                tag_author, lang, fmt, tags = clean_tags(p.get("tags"))
                author = tag_author or title_author
                lang = lang or fallback_lang or guess_language(title)

                description = html_to_text(p.get("body_html"))
                images = p.get("images") or []

                yield make_row(
                    source=site_name(url),
                    title=title,
                    author=author,
                    price=to_price(variant.get("price")),
                    vendor=p.get("vendor"),
                    language=lang,
                    format=fmt,
                    tags=tags,
                    book_url=f"{site_base(url)}/products/{p.get('handle', '')}",
                    image_url=images[0].get("src", "") if images else "",
                    description=description,
                )

    def accept(self, row):
        if not row["title"] or is_junk(row["title"]):
            self.stats["junk"] += 1
            return False
        if row["book_url"] in self.seen:
            self.stats["duplicate"] += 1
            return False
        self.seen.add(row["book_url"])
        return True

    def run(self):
        Path(self.output_csv).parent.mkdir(parents=True, exist_ok=True)
        with open(self.output_csv, "w", newline="", encoding="utf-8-sig") as f:
            writer = csv.DictWriter(f, COLUMNS)
            writer.writeheader()
            for url in self.urls:
                kept = 0
                try:
                    for row in self.shopify(url):
                        if not self.accept(row):
                            continue
                        writer.writerow(row)
                        kept += 1
                        self.stats["kept"] += 1
                        if self.max_per_site is not None and kept >= self.max_per_site:
                            break
                except requests.HTTPError as e:
                    code = e.response.status_code if e.response is not None else "?"
                    print(f"[failed] {url}: HTTP {code} (not a Shopify store, or blocked)")
                except (requests.RequestException, ValueError) as e:
                    print(f"[failed] {url}: {type(e).__name__}: {str(e)[:100]}")
                f.flush()
                print(f"{url}: {kept} books")
        print(f"\nTotal unique books: {self.stats['kept']} -> {self.output_csv}")
        print(f"Skipped: {self.stats['junk']} junk, {self.stats['duplicate']} duplicates, "
              f"{self.stats['out_of_stock']} out of stock")

# Check Data
def check_data(csv_path):
    df = pd.read_csv(csv_path, keep_default_na=False)
    df["price"] = pd.to_numeric(df["price"], errors="coerce")
    print(df.head())
    print("\nShape:", df.shape)
    print("\nUnique values per column:\n", df.nunique())
    print("\nEmpty description share per source:\n",
          (df["description"] == "").groupby(df["source"]).mean())
    return df


if __name__ == "__main__":
    scraper = BookScraper(URLS)
    scraper.run()
    check_data(OUTPUT_CSV)