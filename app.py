# app.py

import os
import torch
from fastapi import FastAPI
from pydantic import BaseModel
from langchain_community.document_loaders import PyPDFLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.vectorstores import Chroma
from langchain_community.llms import Ollama
from langchain.chains import RetrievalQA

# ---------- Embedding Setup ----------
def ingest_pdf(filepath: str):
    if not os.path.exists(filepath):
        print(f"❌ File tidak ditemukan: {filepath}")
        return

    print(f"📄 Memuat dokumen dari {filepath}")
    loader = PyPDFLoader(filepath)
    docs = loader.load()

    print(f"✂️ Memotong dokumen...")
    splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=50)
    docs_split = splitter.split_documents(docs)

    print(f"🧠 Membuat embedding dan menyimpan ke db/...")
    embedding = HuggingFaceEmbeddings(
        model_name="BAAI/bge-m3",
        model_kwargs={"device": "cuda" if torch.cuda.is_available() else "cpu"},
        encode_kwargs={"normalize_embeddings": True}
    )

    vectordb = Chroma.from_documents(docs_split, embedding, persist_directory="db/")
    vectordb.persist()
    print("✅ Embedding selesai.")

# ---------- FastAPI Chat Setup ----------
embedding = HuggingFaceEmbeddings(
    model_name="BAAI/bge-m3",
    model_kwargs={"device": "cuda" if torch.cuda.is_available() else "cpu"},
    encode_kwargs={"normalize_embeddings": True}
)

vectordb = Chroma(persist_directory="db/", embedding_function=embedding)
retriever = vectordb.as_retriever()
llm = Ollama(model="mistral")
qa_chain = RetrievalQA.from_chain_type(llm=llm, retriever=retriever)

app = FastAPI()

class QueryRequest(BaseModel):
    question: str

@app.get("/")
def root():
    return {"message": "QA API aktif 🚀"}

@app.post("/ask")
def ask(req: QueryRequest):
    answer = qa_chain.run(req.question)
    return {"answer": answer}

# ---------- Mode CLI atau API ----------
if __name__ == "__main__":
    import argparse
    import uvicorn

    parser = argparse.ArgumentParser()
    parser.add_argument("--pdf", help="Path PDF untuk ingest")
    parser.add_argument("--api", action="store_true", help="Jalankan sebagai API")
    args = parser.parse_args()

    if args.pdf:
        ingest_pdf(args.pdf)
    elif args.api:
        uvicorn.run("app:app", host="0.0.0.0", port=8000, reload=False)
    else:
        print("❗ Gunakan --pdf [path] untuk ingest atau --api untuk jalankan API")
