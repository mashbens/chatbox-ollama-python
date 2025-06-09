import torch
from langchain_community.vectorstores import Chroma
from langchain_community.embeddings import HuggingFaceEmbeddings

def load_chromadb(persist_dir="db"):
    print("🧠 Memuat model embedding BAAI/bge-m3...")
    embedding = HuggingFaceEmbeddings(
        model_name="BAAI/bge-m3",
        model_kwargs={"device": "cuda" if torch.cuda.is_available() else "cpu"},
        encode_kwargs={"normalize_embeddings": True}
    )

    print(f"💾 Memuat database dari folder '{persist_dir}'...")
    vectordb = Chroma(persist_directory=persist_dir, embedding_function=embedding)

    return vectordb


def tampilkan_dokumen(vectordb, jumlah=5):
    print(f"\n📄 Menampilkan {jumlah} dokumen pertama:")
    data = vectordb.get()
    docs = data["documents"]
    metas = data["metadatas"]

    for i in range(min(jumlah, len(docs))):
        print(f"\n--- Document {i+1} ---")
        print("Isi:", docs[i][:200], "..." if len(docs[i]) > 200 else "")
        print("Metadata:", metas[i])


def cari_dokumen(vectordb, query, category_filter=None, module_filter=None, k=3):
    print(f"\n🔍 Mencari: '{query}'")
    filters = {}

    if category_filter:
        filters["category"] = category_filter
    if module_filter:
        filters["module"] = module_filter

    results = vectordb.similarity_search(query, k=k, filter=filters if filters else None)

    if not results:
        print("❌ Tidak ada hasil ditemukan.")
        return

    for i, doc in enumerate(results):
        print(f"\n--- Hasil {i+1} ---")
        print("Isi:", doc.page_content[:200], "..." if len(doc.page_content) > 200 else "")
        print("Metadata:", doc.metadata)


if __name__ == "__main__":
    db = load_chromadb()

    # Tampilkan dokumen
    tampilkan_dokumen(db, jumlah=5)

    # Contoh pencarian bebas
    cari_dokumen(db, query="apa itu data science?")

    # Contoh pencarian dengan filter category/module
    cari_dokumen(db, query="pelatihan", category_filter="mekaar")
    cari_dokumen(db, query="sistem informasi", module_filter="data-ai")
