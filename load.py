import os
import torch
from langchain_community.document_loaders import PyPDFLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.vectorstores import Chroma


def ingest_pdfs(filepaths: list):
    all_docs = []

    for filepath in filepaths:
        if not os.path.exists(filepath):
            print(f"❌ File tidak ditemukan: {filepath}")
            continue

        print(f"📄 Memuat dokumen dari {filepath}")
        try:
            loader = PyPDFLoader(filepath)
            docs = loader.load()
            for doc in docs:
                doc.metadata["source"] = os.path.basename(filepath)  # Simpan nama file
            all_docs.extend(docs)
        except Exception as e:
            print(f"❌ Terjadi kesalahan saat memuat file PDF {filepath}: {e}")
            continue

    if not all_docs:
        print("❌ Tidak ada dokumen yang dimuat.")
        return

    print("✂️ Memotong dokumen jadi chunks...")
    splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=50)
    docs_split = splitter.split_documents(all_docs)

    print("🧠 Memuat embedding model BAAI/bge-m3...")
    try:
        embedding = HuggingFaceEmbeddings(
            model_name="BAAI/bge-m3",
            model_kwargs={"device": "cuda" if torch.cuda.is_available() else "cpu"},
            encode_kwargs={"normalize_embeddings": True}
        )
    except Exception as e:
        print(f"❌ Gagal memuat embedding model: {e}")
        return

    print("💾 Menyimpan ke ChromaDB (append jika sudah ada)...")
    try:
        vectordb = Chroma(persist_directory="db", embedding_function=embedding)
        vectordb.add_documents(docs_split)
        vectordb.persist()
    except Exception as e:
        print(f"❌ Gagal menyimpan ke ChromaDB: {e}")
        return

    print("✅ Proses selesai. Embedding disimpan ke 'db/'.")


if __name__ == "__main__":
    filepaths = ["./docs/AquaCoolDispenser.pdf", "./docs/SmartCleanVacuum.pdf"]  # Tambah lebih banyak PDF di sini
    ingest_pdfs(filepaths)
