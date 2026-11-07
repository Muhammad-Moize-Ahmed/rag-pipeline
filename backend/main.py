import os
import re
from pathlib import Path
from typing import List, Optional

# --- Memory Optimization for Render 512MB RAM ---
os.environ["OMP_NUM_THREADS"] = "1"
os.environ["MKL_NUM_THREADS"] = "1"
os.environ["TOKENIZERS_PARALLELISM"] = "false"

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse
from pydantic import BaseModel
import chromadb
from chromadb.utils.embedding_functions import ONNXMiniLM_L6_V2
from openai import OpenAI
from pypdf import PdfReader
from dotenv import load_dotenv

# Load local .env if available
load_dotenv()

# --- Absolute Path Configurations ---
# BASE_DIR resolves to root folder: class_project
BASE_DIR = Path(__file__).resolve().parent.parent

CHROMA_PATH = os.getenv("CHROMA_PATH", str(BASE_DIR / "chroma_db"))
PDF_DIR = Path(os.getenv("PDF_DIR", str(BASE_DIR / "data" / "pdfs")))
HTML_PATH = BASE_DIR / "frontend" / "index.html"

# --- Model & Prompt Configurations ---
# Embeddings use chromadb's bundled ONNX build of all-MiniLM-L6-v2 (no torch),
# which keeps memory well under Render's 512MB limit.
LLM_BASE_URL = os.getenv("BASE_URL", "https://generativelanguage.googleapis.com/v1beta/openai/")
LLM_MODEL = os.getenv("LLM_MODEL", "gemini-3.5-flash")
LLM_API_KEY = os.getenv("LLM_API_KEY", "")

EMBED_BATCH = 4
MIN_SCORE = 0.25
NOT_COVERED_MSG = "Sorry, this question is not covered in the course material I have."

SYSTEM_PROMPT = (
    "You are a course assistant for students. Answer ONLY using the numbered context excerpts provided. "
    "Do not use outside knowledge. Add the excerpt number in square brackets, like [1], after each fact you use. "
    "If the excerpts do not contain the answer, reply with exactly: NOT_COVERED"
)

# --- FastAPI Initialization ---
app = FastAPI(title="Course RAG Assistant API", version="1.0.0")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# --- Service State ---
embed_model = None
chroma_client = None
collection = None
llm_client = None

@app.on_event("startup")
def startup_event():
    global embed_model, chroma_client, collection, llm_client
    embed_model = ONNXMiniLM_L6_V2(preferred_providers=["CPUExecutionProvider"])
    chroma_client = chromadb.PersistentClient(path=CHROMA_PATH)
    collection = chroma_client.get_or_create_collection(
        name="course",
        metadata={"hnsw:space": "cosine"}
    )
    llm_client = OpenAI(base_url=LLM_BASE_URL, api_key=LLM_API_KEY or "dummy_key")
    # Render's disk is ephemeral and chroma_db/ is gitignored, so build the index on boot if empty.
    if collection.count() == 0 and PDF_DIR.exists():
        print("Index empty, ingesting PDFs...", ingest_pdfs())

# --- Helper Functions ---
def chunk_text(text: str, size: int = 800, overlap: int = 150) -> List[str]:
    chunks, start = [], 0
    while start < len(text):
        chunks.append(text[start : start + size])
        start += size - overlap
    return chunks

def retrieve(question: str, k: int = 4):
    q_emb = [e.tolist() for e in embed_model([question])]
    r = collection.query(query_embeddings=q_emb, n_results=k)
    
    if not r["documents"] or not r["documents"][0]:
        return []

    return [
        {
            "text": doc,
            "doc": meta.get("doc", "Unknown"),
            "page": meta.get("page", 0),
            "score": round(1 - dist, 3),
        }
        for doc, meta, dist in zip(r["documents"][0], r["metadatas"][0], r["distances"][0])
    ]

# --- Schemas ---
class QueryRequest(BaseModel):
    question: str
    top_k: Optional[int] = 4

class SourceItem(BaseModel):
    doc: str
    page: int
    score: Optional[float] = None

class QueryResponse(BaseModel):
    question: str
    answer: str
    sources: List[SourceItem]
    covered: bool

# --- Endpoints ---
@app.get("/")
def serve_home():
    if not HTML_PATH.exists():
        raise HTTPException(status_code=404, detail=f"Frontend file not found at {HTML_PATH}")
    return FileResponse(HTML_PATH)

@app.get("/health")
def health():
    return {
        "status": "healthy",
        "docs_in_index": collection.count() if collection else 0
    }

@app.post("/query", response_model=QueryResponse)
def query_pipeline(req: QueryRequest):
    if not req.question.strip():
        raise HTTPException(status_code=400, detail="Question cannot be empty.")

    hits = retrieve(req.question, k=req.top_k)
    
    if not hits or hits[0]["score"] < MIN_SCORE:
        return QueryResponse(
            question=req.question,
            answer=NOT_COVERED_MSG,
            sources=[],
            covered=False
        )

    context = "\n\n".join(
        f"[{i+1}] ({h['doc']}, page {h['page']})\n{h['text']}"
        for i, h in enumerate(hits)
    )

    try:
        resp = llm_client.chat.completions.create(
            model=LLM_MODEL,
            temperature=0,
            messages=[
                {"role": "system", "content": SYSTEM_PROMPT},
                {"role": "user", "content": f"Context:\n{context}\n\nQuestion: {req.question}"},
            ],
        )
        text = (resp.choices[0].message.content or "").strip()
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"LLM generation failed: {str(e)}")

    if "NOT_COVERED" in text:
        return QueryResponse(
            question=req.question,
            answer=NOT_COVERED_MSG,
            sources=[],
            covered=False
        )

    used = sorted({int(n) for n in re.findall(r"\[(\d+)\]", text) if 1 <= int(n) <= len(hits)})
    cited = [hits[n - 1] for n in used] or hits[:1]

    seen = set()
    sources = []
    for h in cited:
        key = (h["doc"], h["page"])
        if key not in seen:
            seen.add(key)
            sources.append(SourceItem(doc=h["doc"], page=h["page"], score=h["score"]))

    return QueryResponse(
        question=req.question,
        answer=text,
        sources=sources,
        covered=True
    )

@app.post("/ingest")
def ingest_pdfs():
    """Reads PDF files from PDF_DIR, chunks them, and stores them in ChromaDB."""
    if not PDF_DIR.exists():
        raise HTTPException(status_code=404, detail=f"Directory '{PDF_DIR}' not found.")

    pages = []
    for pdf in PDF_DIR.glob("*.pdf"):
        reader = PdfReader(pdf)
        for i, page in enumerate(reader.pages, start=1):
            text = (page.extract_text() or "").strip()
            if text:
                pages.append({"doc": pdf.name, "page": i, "text": text})

    if not pages:
        return {"message": "No readable pages found in PDFs."}

    chunks = []
    for p in pages:
        for j, c in enumerate(chunk_text(p["text"])):
            chunks.append({
                "id": f"{p['doc']}-p{p['page']}-c{j}",
                "text": c,
                "doc": p["doc"],
                "page": p["page"],
            })

    # Embed in small batches: onnxruntime keeps its peak buffer allocated, so big batches
    # permanently inflate memory (batch 32 ~580MB vs batch 4 ~270MB).
    texts = [c["text"] for c in chunks]
    embeddings = []
    for i in range(0, len(texts), EMBED_BATCH):
        embeddings.extend(e.tolist() for e in embed_model(texts[i : i + EMBED_BATCH]))

    collection.upsert(
        ids=[c["id"] for c in chunks],
        documents=[c["text"] for c in chunks],
        embeddings=embeddings,
        metadatas=[{"doc": c["doc"], "page": c["page"]} for c in chunks],
    )

    return {
        "status": "success",
        "total_chunks_indexed": len(chunks),
        "total_in_collection": collection.count()
    }

if __name__ == "__main__":
    import uvicorn
    port = int(os.environ.get("PORT", 8000))
    uvicorn.run("main:app", host="0.0.0.0", port=port, reload=False)
