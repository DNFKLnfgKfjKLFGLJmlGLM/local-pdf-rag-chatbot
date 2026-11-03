# Local PDF RAG Chatbot — Version 2

A ready-to-run, API-key-free PDF RAG chatbot using:

- Python
- Streamlit
- PyMuPDF
- Sentence Transformers
- ChromaDB
- Ollama

## 1. Install Ollama

Install Ollama from https://ollama.com/

Then download a local model:

    ollama pull llama3.2

Make sure Ollama is running.

## 2. Create a virtual environment

Windows:

    python -m venv .venv
    .venv\Scripts\activate

macOS/Linux:

    python3 -m venv .venv
    source .venv/bin/activate

## 3. Install Python packages

    pip install -r requirements.txt

## 4. Run

    streamlit run app.py

## 5. Use

1. Open the Streamlit page.
2. Select one or more PDFs.
3. Click "Index uploaded PDFs".
4. Ask questions in the chat box.
5. Expand "Sources" to see PDF filenames and page numbers.

## Notes

- The first use of the embedding model downloads it once from Hugging Face.
  After that, embeddings run locally.
- Ollama runs the LLM locally.
- No OpenAI API key is required.
- This version works best with PDFs containing selectable text.
- Scanned/image-only PDFs need OCR before indexing.
- The Chroma database is stored locally in `chroma_db/`.
