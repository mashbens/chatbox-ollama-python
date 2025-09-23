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
    model_kwargs={"device": "cpu"},
    encode_kwargs={"normalize_embeddings": True}
)

# --- Koneksi ke Qdrant ---
print("[INIT] Menghubungkan ke Qdrant...")
client = QdrantClient(host="qdrant", port=6333)

# --- Inisialisasi VectorDB ---
collection_name = "pdf_collection"
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

        llm = OllamaLLM(model="llama3.1:8b", base_url="http://ollama:11434")

        # Query ke Qdrant
        print("[INFO] Mencari dokumen relevan dari Qdrant...")
        filter_query = {"module": module} if module else None
        all_docs = vectordb.similarity_search_with_score(question, k=15, filter=filter_query)

        # Filter dan simpan (doc, score)
        threshold = 0.7
        filtered_docs = []
        for doc, score in all_docs:
            if score > threshold:
                print(f"[FILTER] Dibuang karena score terlalu tinggi. Score: {score:.4f} | Page: {doc.metadata.get('page_number')} | Source: {doc.metadata.get('source')}")
                continue
            if module and f"{module}.pdf" not in doc.metadata.get("source", ""):
                print(f"[FILTER] Dibuang karena tidak sesuai modul. Score: {score:.4f} | Page: {doc.metadata.get('page_number')} | Source: {doc.metadata.get('source')}")
                continue
            print(f"[KEEP] Dipertahankan. Score: {score:.4f} | Page: {doc.metadata.get('page_number')} | Source: {doc.metadata.get('source')}")
            filtered_docs.append((doc, score))

        if not filtered_docs:
            print("[INFO] Tidak ada dokumen relevan yang ditemukan.")
            return jsonify({
                "response": {
                    "jawaban": "Maaf, saya tidak menemukan jawaban untuk pertanyaan tersebut dalam modul-modul PNM yang diberikan.",
                    "sumber": []
                }
            })

        # Urutkan berdasarkan page_number
        filtered_docs.sort(key=lambda x: int(x[0].metadata.get('page_number', 0)))

        # Susun context dan sumber
        print(f"[INFO] {len(filtered_docs)} dokumen lolos filter. Menyusun konteks...")
        context_parts = []
        sumber_set = set()

        for doc, score in filtered_docs:
            page = doc.metadata.get('page_number', '?')
            source = doc.metadata.get('source', 'unknown')
            context_parts.append(f"[Halaman {page}] {doc.page_content.strip()}")
            sumber_set.add(f"{source} (halaman {page})")
            print(f"[CONTEXT] Score: {score:.4f} | Page: {page} | Source: {source}")

        context = "\n\n".join(context_parts)
        sumber_list = sorted(sumber_set, key=lambda s: int(s.split("halaman ")[-1].rstrip(")")))

        # Prompt
# Anda adalah asisten AI yang membantu menjawab pertanyaan berdasarkan dokumen resmi PNM. Jawab hanya berdasarkan informasi yang terdapat dalam dokumen tersebut, dan jangan mengarang.
        prompt = f"""
Saya adalah AI PNM. Jika informasi tidak ada dalam dokumen, katakan "Maaf, saya tidak menemukan jawaban untuk pertanyaan tersebut dalam modul-modul PNM yang diberikan."
Berikut adalah potongan dokumen yang relevan:
{context}
Berdasarkan isi dokumen tersebut, ringkas dan jelaskan informasi penting yang terkandung di dalamnya.

Jika memungkinkan, susun jawaban dalam format poin-poin sebagai berikut:
1. [Judul Topik atau Kebijakan]
2. [Penjelasan singkat]
3. [Contoh atau penerapan](optional jika ada)

Di akhir cantumkan Sumber dan halaman dokumen.
Jika tidak ditemukan kebijakan, peraturan, atau prosedur tertentu, cukup berikan penjelasan umum yang sesuai dengan isi dokumen.
Tolong jawab dengan bahasa yang jelas, padat, dan terstruktur.

Pertanyaan:
{question}

Jawaban:
"""

        print("[INFO] Mengirim prompt ke LLM...")
        # jawaban = llm.invoke(prompt)
        print("[INFO] Jawaban LLM diterima.")

        return jsonify({
            "response": {
                # "jawaban": jawaban.strip(),
                "jawaban": prompt.strip(),
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
