from flask import Flask, request, jsonify
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_ollama import OllamaLLM
from langchain_qdrant import Qdrant
from qdrant_client import QdrantClient
import traceback

# --- Konfigurasi Flask ---
app = Flask(__name__)

# --- Konfigurasi Embedding ---
print("[INIT] Memuat embedding model...")
embedding = HuggingFaceEmbeddings(
    model_name="BAAI/bge-m3",
    model_kwargs={"device": "cpu"},  # Ganti ke 'cuda' jika pakai GPU
    encode_kwargs={"normalize_embeddings": True}
)

# --- Koneksi ke Qdrant ---
print("[INIT] Menghubungkan ke Qdrant...")
client = QdrantClient(host="qdrant", port=6333)  

# --- Inisialisasi VectorDB ---
collection_name = "pdf_collection"  # Pastikan sama dengan yang digunakan saat ingest
vectordb = Qdrant(
    client=client,
    collection_name=collection_name,
    embeddings=embedding
)

print("[INIT] Vector DB Qdrant berhasil dimuat.")

# --- Endpoint Chat ---
@app.route("/ask", methods=["POST"])
def chat():
    data = request.get_json()
    question = data.get("question", "").strip()
    module = data.get("module", "").strip()

    if not question:
        return jsonify({"error": "Parameter 'question' tidak boleh kosong."}), 400

    try:
        print("========== NEW REQUEST ==========")
        print("Pertanyaan:", question)
        print("Module filter:", module)

        # Load LLM
        llm = OllamaLLM(model="pnm-mistral", base_url="http://ollama:11434")

        # Query ke Qdrant
        print("[INFO] Mencari dokumen relevan dari Qdrant...")
        filter_query = {"module": module} if module else None
        all_docs = vectordb.similarity_search_with_score(question, k=5, filter=filter_query)

        print("[DEBUG] Dokumen hasil similarity_search_with_score:")
        for i, (doc, score) in enumerate(all_docs):
            print(f"  {i+1}. Score: {score:.4f} | Source: {doc.metadata.get('source', 'unknown')}")

        # Filter dokumen
        threshold = 1.3
        filtered_docs = []
        for doc, score in all_docs:
            source = doc.metadata.get("source", "")
            if score > threshold:
                print(f"[FILTER] Dokumen dengan score {score:.4f} dibuang karena > threshold {threshold}")
                continue
            if module and f"{module}.pdf" not in source:
                print(f"[FILTER] Dokumen '{source}' dibuang karena bukan dari modul {module}")
                continue
            filtered_docs.append(doc)

        if not filtered_docs:
            print("[INFO] Tidak ada dokumen relevan yang ditemukan.")
            return jsonify({
                "response": {
                    "jawaban": "Maaf, saya tidak menemukan jawaban untuk pertanyaan tersebut dalam modul-modul PNM yang diberikan.",
                    "sumber": []
                }
            })

        print(f"[INFO] {len(filtered_docs)} dokumen lolos filter. Siap dibuat prompt.")
        context = "\n\n".join([doc.page_content for doc in filtered_docs])
        sumber_list = list({doc.metadata.get("source", "unknown") for doc in filtered_docs})

        print("[DEBUG] Daftar sumber dokumen:", sumber_list)

        # Buat prompt
        prompt = f"""
        Jika pertanyaan terlalu singkat, tidak jelas, atau tidak memiliki konteks yang cukup untuk dijawab berdasarkan isi modul di bawah ini, mohon jawab:
        "Pertanyaan Anda terlalu umum. Mohon ajukan pertanyaan yang lebih spesifik agar saya bisa membantu dengan tepat."


        Pertanyaan: {question}

        Isi modul:
        {context}

        Jawaban:
        """

        print("[INFO] Mengirim prompt ke LLM...")
        jawaban = llm.invoke(prompt)
        print("[INFO] Jawaban LLM diterima.")

        if sumber_list == ["data-ai.pdf"]:
            sumber_list = []

        return jsonify({
            "response": {
                "jawaban": jawaban.strip(),
                "sumber": sumber_list
            }
        })

    except Exception as e:
        error_message = str(e)
        traceback_str = traceback.format_exc()
        print("Terjadi error:", error_message)
        print(traceback_str)
        return jsonify({
            "error": "Terjadi kesalahan saat memproses permintaan.",
            "detail": error_message,
            "trace": traceback_str
        }), 500

if __name__ == "__main__":
    print("[SERVER] Aplikasi Flask berjalan di http://0.0.0.0:5000")
    app.run(debug=True, host="0.0.0.0", port=5000)
