import os
import torch
from langchain_community.document_loaders import PyPDFLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.vectorstores import Chroma


def ingest_pdfs(file_list: list[dict]):
    all_docs = []

    for file in file_list:
        filepath = file["path"]
        module = file["module"]
        category = file.get("category", "other")

        if not os.path.exists(filepath):
            print(f"❌ File tidak ditemukan: {filepath}")
            continue

        print(f"📄 Memuat dokumen dari {filepath}")
        try:
            loader = PyPDFLoader(filepath)
            docs = loader.load()

            for doc in docs:
                doc.metadata["source"] = f"{module}.pdf"     # Untuk filter di API
                doc.metadata["filepath"] = filepath
                doc.metadata["category"] = category
                doc.metadata["module"] = module              # Simpan juga tanpa ".pdf" kalau butuh

            all_docs.extend(docs)
        except Exception as e:
            print(f"❌ Gagal memuat {filepath}: {e}")
            continue

    if not all_docs:
        print("❌ Tidak ada dokumen yang berhasil dimuat.")
        return

    print("✂️ Memotong dokumen menjadi chunks...")
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
        print(f"❌ Gagal memuat model embedding: {e}")
        return

    print("💾 Menyimpan ke ChromaDB (append jika sudah ada)...")
    try:
        vectordb = Chroma(persist_directory="db", embedding_function=embedding)
        vectordb.add_documents(docs_split)
        vectordb.persist()
    except Exception as e:
        print(f"❌ Gagal menyimpan ke ChromaDB: {e}")
        return

    print("✅ Sukses! Embedding disimpan ke folder 'db/'.")


if __name__ == "__main__":
    file_list = [
        {
            "path": "./docs/PPM.pdf",
            "module": "mekaar",
            "category": "mekaar"
        },
        {
            "path": "./docs/PPU.pdf",
            "module": "ulamm",
            "category": "ulamm"
        }
        # Tambahkan file lainnya di sini
    ]
    ingest_pdfs(file_list)
