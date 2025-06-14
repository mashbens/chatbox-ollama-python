import os
import torch
from flask import Flask, request, jsonify
from werkzeug.utils import secure_filename

from langchain_community.document_loaders import PyPDFLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_qdrant import Qdrant

from qdrant_client import QdrantClient
from qdrant_client.models import Distance, VectorParams

# --- Konfigurasi Flask App ---
app = Flask(__name__)
ALLOWED_EXTENSIONS = {'pdf'}

BASE_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
UPLOAD_FOLDER = os.path.join(BASE_DIR, "uploads")

# UPLOAD_FOLDER = "/app/uploads" 

app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER
os.makedirs(UPLOAD_FOLDER, exist_ok=True)

# --- Fungsi Validasi File ---
def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

# --- Fungsi Ingest PDF ---
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
                doc.metadata["source"] = f"{module}.pdf"
                doc.metadata["filepath"] = filepath
                doc.metadata["category"] = category
                doc.metadata["module"] = module

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

    print("🧠 Memuat model embedding BAAI/bge-m3...")
    try:
        embedding = HuggingFaceEmbeddings(
            model_name="BAAI/bge-m3",
            model_kwargs={"device": "cuda" if torch.cuda.is_available() else "cpu"},
            encode_kwargs={"normalize_embeddings": True}
        )
    except Exception as e:
        print(f"❌ Gagal memuat model embedding: {e}")
        return

    print("💾 Menyimpan ke Qdrant...")
    try:
        client = QdrantClient(
            host="qdrant",  # sesuaikan dengan nama service di docker-compose
            port=6333,
        )

        collection_name = "pdf_collection"

        # Buat collection jika belum ada
        if not client.collection_exists(collection_name):
            client.create_collection(
                collection_name=collection_name,
                vectors_config=VectorParams(
                    size=1024,
                    distance=Distance.COSINE
                ),
            )

        vectordb = Qdrant(
            client=client,
            collection_name=collection_name,
            embeddings=embedding,
        )

        vectordb.add_documents(docs_split)
        print("✅ Sukses! Data tersimpan ke Qdrant.")
    except Exception as e:
        print(f"❌ Gagal menyimpan ke Qdrant: {e}")
        return

# --- Endpoint Upload PDF ---
@app.route("/upload", methods=["POST"])
def upload_file():
    if 'file' not in request.files:
        return jsonify({"error": "No file uploaded"}), 400

    file = request.files['file']
    module = request.form.get('module')
    category = request.form.get('category', 'other')

    if not file or file.filename == '':
        return jsonify({"error": "No file selected"}), 400

    if not allowed_file(file.filename):
        return jsonify({"error": "Invalid file type"}), 400

    if not module:
        return jsonify({"error": "Module name is required"}), 400

    filename = secure_filename(file.filename)
    filepath = os.path.join(app.config['UPLOAD_FOLDER'], filename)
    file.save(filepath)

    print(f"✅ File disimpan di: {filepath}")

    file_info = [{
        "path": filepath,
        "module": module,
        "category": category
    }]

    ingest_pdfs(file_info)

    return jsonify({"message": "File uploaded and ingested successfully!"})

# --- Run App ---
if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5001)
