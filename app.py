import os
import hashlib
import requests
import fitz
import chromadb
import streamlit as st
from sentence_transformers import SentenceTransformer

APP_TITLE = "Local PDF RAG Chatbot"
OLLAMA_URL = "http://localhost:11434/api/generate"
MODEL_NAME = "llama3.2"
DB_DIR = "chroma_db"
COLLECTION_NAME = "pdf_rag_v2"
EMBED_MODEL = "all-MiniLM-L6-v2"

st.set_page_config(page_title=APP_TITLE, page_icon="📚", layout="wide")

@st.cache_resource
def get_embedding_model():
    return SentenceTransformer(EMBED_MODEL)

@st.cache_resource
def get_collection():
    client = chromadb.PersistentClient(path=DB_DIR)
    return client.get_or_create_collection(name=COLLECTION_NAME)

embedding_model = get_embedding_model()
collection = get_collection()

def file_hash(data):
    return hashlib.sha256(data).hexdigest()[:16]

def chunk_text(text, chunk_words=180, overlap=40):
    words = text.split()
    chunks = []
    step = max(1, chunk_words - overlap)
    for start in range(0, len(words), step):
        chunk = " ".join(words[start:start + chunk_words]).strip()
        if len(chunk) >= 40:
            chunks.append(chunk)
        if start + chunk_words >= len(words):
            break
    return chunks

def index_pdf(pdf_bytes, filename):
    digest = file_hash(pdf_bytes)
    existing = collection.get(where={"file_hash": digest}, limit=1)
    if existing["ids"]:
        return 0, True

    doc = fitz.open(stream=pdf_bytes, filetype="pdf")
    documents, metadatas, ids = [], [], []

    for page_no, page in enumerate(doc, start=1):
        text = page.get_text("text")
        for chunk_no, chunk in enumerate(chunk_text(text)):
            documents.append(chunk)
            metadatas.append({
                "source": filename,
                "page": page_no,
                "file_hash": digest
            })
            ids.append(f"{digest}_{page_no}_{chunk_no}")

    if not documents:
        raise ValueError("No selectable text was found. This PDF may be scanned/image-only.")

    vectors = embedding_model.encode(
        documents, normalize_embeddings=True, show_progress_bar=False
    ).tolist()

    collection.add(
        ids=ids,
        documents=documents,
        metadatas=metadatas,
        embeddings=vectors
    )
    return len(documents), False

def search(question, top_k=5):
    vector = embedding_model.encode(
        [question], normalize_embeddings=True
    ).tolist()
    result = collection.query(
        query_embeddings=vector,
        n_results=top_k
    )
    docs = result.get("documents", [[]])[0]
    metas = result.get("metadatas", [[]])[0]
    return list(zip(docs, metas))

def ask_ollama(question, retrieved):
    context = "\n\n".join(
        f"[Source: {m['source']}, page {m['page']}]\n{d}"
        for d, m in retrieved
    )

    prompt = f"""You are a careful PDF question-answering assistant.

Use ONLY the supplied context. If the context does not contain
the answer, say: "I could not find this information in the uploaded PDFs."

Do not invent facts. Keep the answer clear and useful.
When useful, mention the source page naturally.

CONTEXT:
{context}

QUESTION:
{question}

ANSWER:
"""

    r = requests.post(
        OLLAMA_URL,
        json={"model": MODEL_NAME, "prompt": prompt, "stream": False},
        timeout=180
    )
    r.raise_for_status()
    return r.json()["response"].strip()

def indexed_files():
    data = collection.get(include=["metadatas"])
    names = sorted({
        m.get("source")
        for m in data.get("metadatas", [])
        if m and m.get("source")
    })
    return names

st.title("📚 Local PDF RAG Chatbot")
st.caption("Multiple PDFs • Local embeddings • Local LLM • Page citations • No API key")

with st.sidebar:
    st.header("1. Upload PDFs")
    uploads = st.file_uploader(
        "Select one or more PDF files",
        type=["pdf"],
        accept_multiple_files=True
    )

    if st.button("📥 Index uploaded PDFs", use_container_width=True):
        if not uploads:
            st.warning("Choose at least one PDF.")
        else:
            progress = st.progress(0)
            for i, uploaded in enumerate(uploads):
                try:
                    n, duplicate = index_pdf(uploaded.getvalue(), uploaded.name)
                    if duplicate:
                        st.info(f"{uploaded.name}: already indexed.")
                    else:
                        st.success(f"{uploaded.name}: indexed {n} chunks.")
                except Exception as e:
                    st.error(f"{uploaded.name}: {e}")
                progress.progress((i + 1) / len(uploads))

    st.header("2. Indexed documents")
    names = indexed_files()
    if names:
        for name in names:
            st.write(f"📄 {name}")
    else:
        st.caption("No PDFs indexed yet.")

    st.divider()
    st.caption(f"Embedding model: {EMBED_MODEL}")
    st.caption(f"LLM: Ollama / {MODEL_NAME}")

if "messages" not in st.session_state:
    st.session_state.messages = []

for msg in st.session_state.messages:
    with st.chat_message(msg["role"]):
        st.markdown(msg["content"])
        if msg.get("sources"):
            with st.expander("📖 Sources"):
                for s in msg["sources"]:
                    st.markdown(f"- **{s['source']}**, page **{s['page']}**")

question = st.chat_input("Ask something about your uploaded PDFs...")

if question:
    st.session_state.messages.append({"role": "user", "content": question})
    with st.chat_message("user"):
        st.markdown(question)

    with st.chat_message("assistant"):
        if collection.count() == 0:
            st.warning("Please upload and index at least one PDF first.")
        else:
            with st.spinner("Searching your PDFs..."):
                retrieved = search(question, top_k=5)

            if not retrieved:
                answer = "I could not find this information in the uploaded PDFs."
            else:
                with st.spinner("Generating a local answer..."):
                    try:
                        answer = ask_ollama(question, retrieved)
                    except requests.exceptions.ConnectionError:
                        answer = "Ollama is not running. Start Ollama and try again."
                    except Exception as e:
                        answer = f"Could not generate the answer: {e}"

            st.markdown(answer)

            sources = []
            seen = set()
            for _, meta in retrieved:
                key = (meta["source"], meta["page"])
                if key not in seen:
                    seen.add(key)
                    sources.append({"source": meta["source"], "page": meta["page"]})

            if sources:
                with st.expander("📖 Sources"):
                    for s in sources:
                        st.markdown(f"- **{s['source']}**, page **{s['page']}**")

            st.session_state.messages.append({
                "role": "assistant",
                "content": answer,
                "sources": sources
            })
