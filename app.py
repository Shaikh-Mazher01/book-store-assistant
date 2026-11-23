import os
from pathlib import Path

import streamlit as st
from dotenv import load_dotenv
from langchain_core.documents import Document
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate

from kitab.config import DEFAULT_DB_DIR, DOCS_JSONL, RERANKER_MODEL
from kitab.query import describe_filters, parse
from kitab.retriever import (HybridRetriever, build_index, get_embeddings,
                             load_documents, open_store)

load_dotenv()

GROQ_MODEL = os.getenv("GROQ_MODEL", "qwen/qwen3.8-27b")

TOP_K = 5  #books retrieved per question

EXAMPLES = [
    "Hindi books on Gandhi under 300",
    "Premchand kahaniyan under 200",
    "Hindi Chanakya books under 250",
    "Marathi Chanakya",
    "Romeo and Juliet",
]


# Cached resources 
@st.cache_resource(show_spinner="Loading the book catalog (first run may takes few minutes)...")
def load_catalog():
    db_dir = Path(os.getenv("BOOKS_DB_DIR") or DEFAULT_DB_DIR)
    embeddings = get_embeddings()
    store = open_store(db_dir, embeddings)

    if store._collection.count() == 0:
        if not DOCS_JSONL.exists():
            return None
        build_index(load_documents(DOCS_JSONL), db_dir, embeddings)

    raw = store.get(include=["documents", "metadatas"])
    docs = [Document(page_content=t, metadata=m) for t, m in zip(raw["documents"], raw["metadatas"])]

    prices = [d.metadata["price"] for d in docs if d.metadata.get("price") is not None]
    snapshots = sorted({d.metadata.get("scraped_at", "") for d in docs} - {""})
    return {
        "retriever": HybridRetriever(store, docs, reranker_loader=load_reranker),
        "count": len(docs),
        "sources": sorted({d.metadata.get("source", "") for d in docs} - {""}),
        "price_min": int(min(prices)) if prices else 0,
        "price_max": int(max(prices)) + 1 if prices else 5000,
        "scraped_at": snapshots[-1] if snapshots else "",
    }


@st.cache_resource(show_spinner="Loading reranker...")
def load_reranker():
    from sentence_transformers import CrossEncoder
    return CrossEncoder(RERANKER_MODEL)


@st.cache_resource
def load_llm():
    if not os.getenv("GROQ_API_KEY"):
        return None
    from langchain_groq import ChatGroq
    return ChatGroq(model=GROQ_MODEL, temperature=0.2)


# Prompt
SYSTEM_PROMPT = """You are Kitab, a book assistant for an Indian online bookstore catalog \
covering English, Hindi, Urdu and Marathi books.
Rules:
- Use ONLY the books listed in CONTEXT. Never invent titles, authors, prices, links or plot details.
- For each recommendation give: title, author (if known), language, price and store, \
a one-line reason it fits, and its link.
- If a book is also sold elsewhere (Other prices), mention the cheapest option.
- Reply in the same language as the question. Keep book titles as written in CONTEXT.
- If nothing in CONTEXT fits, say "I couldn't find a good match in the catalog" and suggest \
how to refine the search (different keywords, language or price range).

CONTEXT:
{context}"""

PROMPT = ChatPromptTemplate.from_messages([("system", SYSTEM_PROMPT), ("human", "{question}")])


def format_books_context(docs: list[Document]) -> str:
    blocks = []
    for i, d in enumerate(docs, 1):
        m = d.metadata
        blocks.append(
            f"[{i}] {d.page_content}\n"
            f"Price: Rs {m.get('price', 0):g} at {m.get('source')}\n"
            f"Other prices: {m.get('all_prices') or 'n/a'}\n"
            f"Link: {m.get('url')}"
        )
    return "\n\n".join(blocks)


# UI helpers
def render_books(books: list[dict]):
    for m in books:
        img_col, txt_col = st.columns([1, 5], vertical_alignment="center")
        with img_col:
            if m.get("image_url"):
                try:
                    st.image(m["image_url"], width=64)
                except Exception:
                    pass
        with txt_col:
            title = m.get("title", "Untitled")
            link = m.get("url")
            st.markdown(f"**[{title}]({link})**" if link else f"**{title}**")
            bits = [m.get("language", "Unknown"), f"Rs {m.get('price', 0):g}", m.get("source", "")]
            if m.get("author"):
                bits.insert(0, m["author"])
            st.caption("  |  ".join(b for b in bits if b))
            if m.get("num_sources", 1) > 1 and m.get("all_prices"):
                st.caption(f"Also at: {m['all_prices']}")


def build_filters(parsed: dict, sb: dict) -> dict:
    f = dict(parsed)
    if sb["sources"]:
        f["sources"] = list(sb["sources"])
    if sb["price"] != sb["price_full"]:
        f["min_price"], f["max_price"] = float(sb["price"][0]), float(sb["price"][1])
    return f


# App
st.set_page_config(page_title="Friendly Book Suggestion", page_icon="📚", layout="centered")

try:
    catalog = load_catalog()
except Exception as e:  # e.g. read-only disk while building the index
    st.error(f"Could not open the book database: {type(e).__name__}: {str(e)[:200]}")
    st.stop()
if catalog is None:
    st.error("Book data not found. Run `python scripts/data_clean.py` and "
             "`python scripts/build_index.py`, or set BOOKS_DB_DIR in your .env file.")
    st.stop()

retriever: HybridRetriever = catalog["retriever"]
llm = load_llm()

# ---- sidebar ----
with st.sidebar:
    st.header("Filters")
    st.caption("Filter with the price range or stores")
    sb_sources = st.multiselect("Store", catalog["sources"])
    price_full = (catalog["price_min"], catalog["price_max"])
    sb_price = st.slider("Price (Rs)", price_full[0], price_full[1], price_full, step=10)

    st.header("Search")
    
    use_rerank = st.toggle("Rerank results (slower, more accurate)", value=False)
    show_books = True if llm is None else st.toggle("Show matching books", value=True)

    if st.button("Clear chat", use_container_width=True):
        st.session_state.messages = []
        st.rerun()
    st.caption(f"{catalog['count']:,} books indexed")
    if catalog["scraped_at"]:
        st.caption(f"Catalog snapshot from {catalog['scraped_at']}. "
                   "Prices and stock may have changed since. so better to check the link after retrival")

sidebar = {"sources": sb_sources, "price": sb_price, "price_full": price_full}

# ---- header ----
st.title("Books Search Assistant")
st.write("Find books in English, Hindi, Urdu and Marathi across Indian online bookstores. "
         "Ask in any of these languages, and range a price or store if you like.")

if llm is None:
    st.warning("GROQ_API_KEY is not set, so you will see matching books but no written answer. "
               "Add it to your .env file to turn answers on.")

if "messages" not in st.session_state:
    st.session_state.messages = []

# ---- history ----
for msg in st.session_state.messages:
    with st.chat_message(msg["role"]):
        st.markdown(msg["content"])
        if msg.get("books") and show_books:
            with st.expander(f"Matching books ({len(msg['books'])})"):
                render_books(msg["books"])

# ---- example questions (only on an empty chat) ----
if not st.session_state.messages:
    st.caption("Try one of these:")
    for ex in EXAMPLES:
        st.button(ex, key=f"ex_{ex}", use_container_width=True,
                  on_click=lambda e=ex: st.session_state.update(pending=e))

question = st.chat_input("Search for book 'Panch Panchayat Aur Panchayati Raj'")
if "pending" in st.session_state:
    question = st.session_state.pop("pending")

# ---- answer ----
if question:
    st.session_state.messages.append({"role": "user", "content": question})
    with st.chat_message("user"):
        st.markdown(question)

    with st.chat_message("assistant"):
        parsed, cleaned = parse(question, sources=catalog["sources"])
        filters = build_filters(parsed, sidebar)

        with st.spinner("Searching the catalog..."):
            docs = retriever.invoke(cleaned, filters, k=TOP_K, rerank=use_rerank)

        applied = describe_filters(filters)
        if applied:
            st.caption(f"Filters applied: {applied}")

        books = [d.metadata for d in docs]
        if not docs:
            reply = ("I couldn't find a good match in the catalog. Try different keywords, "
                     "a wider price range, or remove the language or store filter.")
            st.markdown(reply)
        elif llm is None:
            reply = "Here are the closest matches I found."
            st.markdown(reply)
        else:
            chain = PROMPT | llm | StrOutputParser()
            try:
                reply = st.write_stream(
                    chain.stream({"question": question, "context": format_books_context(docs)})
                )
            except Exception as e:
                reply = "I found matching books but couldn't generate a written answer."
                st.markdown(reply)
                st.error(f"LLM error: {type(e).__name__}: {str(e)[:200]}")

        if books and show_books:
            with st.expander(f"Matching books ({len(books)})", expanded=llm is None):
                render_books(books)

    st.session_state.messages.append({"role": "assistant", "content": reply, "books": books})
