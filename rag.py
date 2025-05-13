# rag.py

import os
import glob
import requests
from pathlib import Path
from dotenv import load_dotenv
from langchain_community.document_loaders import PyPDFLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS

# ─── 1) Load Azure credentials ─────────────────────────────────────────────────
load_dotenv()  # reads .env in this folder

AZURE_BASE    = os.getenv("AZURE_OPENAI_API_BASE", "")
AZURE_KEY     = os.getenv("AZURE_OPENAI_API_KEY", "")
API_VERSION   = os.getenv("AZURE_OPENAI_API_VERSION", "2023-05-15")
DEPLOYMENT_ID = os.getenv("DEPLOYMENT_ID", "gpt-4o")

# ─── 2) Index configuration ─────────────────────────────────────────────────────
PDF_DIR       = Path("raw")
INDEX_DIR     = Path("faiss_store")
CHUNK_SIZE    = 1000
CHUNK_OVERLAP = 200
EMBED_MODEL   = "all-MiniLM-L6-v2"

def build_faiss_index():
    """Load PDFs, split into chunks, embed them, and save a FAISS index."""
    # if index already exists, skip
    if INDEX_DIR.exists() and any(INDEX_DIR.iterdir()):
        return

    # gather all PDFs
    pdf_paths = sorted(glob.glob(str(PDF_DIR / "*.pdf")))
    splitter  = RecursiveCharacterTextSplitter(
        chunk_size=CHUNK_SIZE, chunk_overlap=CHUNK_OVERLAP
    )
    chunks = []

    # split each PDF into chunks
    for pdf_path in pdf_paths:
        docs = PyPDFLoader(pdf_path).load()  # one Document per page
        chunks.extend(splitter.split_documents(docs))

    print(f"🔖 Indexed {len(chunks)} chunks from {len(pdf_paths)} manuals")

    # embed & persist
    emb     = HuggingFaceEmbeddings(model_name=EMBED_MODEL)
    vectordb = FAISS.from_documents(chunks, emb)
    vectordb.save_local(str(INDEX_DIR))
    print(f"💾 FAISS index saved to '{INDEX_DIR}'")

# build index on first import
build_faiss_index()

# ─── 3) Reload FAISS index ──────────────────────────────────────────────────────
emb      = HuggingFaceEmbeddings(model_name=EMBED_MODEL)
vectordb = FAISS.load_local(
    str(INDEX_DIR),
    emb,
    allow_dangerous_deserialization=True
)
print("✅ FAISS index loaded with", len(vectordb.docstore._dict), "chunks")

# ─── 4) Define the RAG function ─────────────────────────────────────────────────
def rag_answer(question: str, k: int = 4) -> str:
    """
    Retrieve the top-k context chunks from FAISS and 
    call Azure GPT-4o via REST to get a step-by-step answer.
    """
    # fetch relevant chunks
    retriever = vectordb.as_retriever(search_kwargs={"k": k})
    docs      = retriever.get_relevant_documents(question)

    # assemble messages
    system_msg = "You are a certified Royal Enfield mechanic. Answer clearly and step-by-step."
    messages   = [{"role": "system", "content": system_msg}]
    for idx, doc in enumerate(docs, start=1):
        messages.append({
            "role":    "assistant",
            "content": f"Context {idx}:\n{doc.page_content}"
        })
    messages.append({"role": "user", "content": question})

    # call Azure OpenAI REST endpoint
    endpoint = f"{AZURE_BASE}/openai/deployments/{DEPLOYMENT_ID}/chat/completions"
    params   = {"api-version": API_VERSION}
    headers  = {"Content-Type": "application/json", "api-key": AZURE_KEY}

    resp = requests.post(endpoint, headers=headers, params=params, json={"messages": messages})
    resp.raise_for_status()
    return resp.json()["choices"][0]["message"]["content"]

# ─── 5) Quick smoke-test when run directly ────────────────────────────────────
if __name__ == "__main__":
    test_q = "How do I bleed the front brake on a Royal Enfield Classic 350?"
    print("Q:", test_q)
    print("A:", rag_answer(test_q))
