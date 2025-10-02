from flask import Flask, request, jsonify, Response, stream_with_context
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_ollama import OllamaLLM
from langchain_qdrant import Qdrant
from qdrant_client import QdrantClient
import traceback
import json

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

# --- Endpoint Chat Normal ---
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

        llm = OllamaLLM(model="pnm-mistral:latest", base_url="http://ollama:11434")

        # Query ke Qdrant
        filter_query = {"module": module} if module else None
        all_docs = vectordb.similarity_search_with_score(question, k=15, filter=filter_query)

        # Filtering
        threshold = 0.7
        filtered_docs = []
        for doc, score in all_docs:
            if score > threshold:
                continue
            if module and f"{module}.pdf" not in doc.metadata.get("source", ""):
                continue
            filtered_docs.append((doc, score))

        if not filtered_docs:
            return jsonify({
                "response": {
                    "jawaban": "Maaf, saya tidak menemukan jawaban untuk pertanyaan tersebut dalam modul-modul PNM.",
                    "sumber": []
                }
            })

        # Urutkan berdasarkan page_number
        filtered_docs.sort(key=lambda x: int(x[0].metadata.get('page_number', 0)))

        # Context + sumber
        context_parts = []
        sumber_set = set()
        for doc, score in filtered_docs:
            page = doc.metadata.get('page_number', '?')
            source = doc.metadata.get('source', 'unknown')
            context_parts.append(f"[Halaman {page}] {doc.page_content.strip()}")
            sumber_set.add(f"{source} (halaman {page})")

        context = "\n\n".join(context_parts)
        sumber_list = sorted(sumber_set, key=lambda s: int(s.split("halaman ")[-1].rstrip(")")))

        prompt = f"""
        Kamu adalah Sabrina, asisten AI ramah yang membantu menjawab pertanyaan berdasarkan dokumen resmi PNM.

        Pertanyaan user:
        {question}

        Dokumen relevan:
        {context}

        Jawaban:
        """

        jawaban = llm.invoke(prompt)

        return jsonify({
            "response": {
                "jawaban": jawaban.strip(),
                "sumber": sumber_list
            }
        })

    except Exception as e:
        return jsonify({
            "error": "Terjadi kesalahan saat memproses permintaan.",
            "detail": str(e),
            "trace": traceback.format_exc()
        }), 500


# --- Endpoint Chat Streaming ---
@app.route("/ask_stream", methods=["POST"])
def chat_stream():
    data = request.get_json()
    question = data.get("question", "").strip()
    module = data.get("module", "").strip()

    if not question:
        return jsonify({"error": "Parameter 'question' tidak boleh kosong."}), 400

    try:
        print("========== NEW STREAM REQUEST ==========")
        print("Pertanyaan:", question)
        print("Module filter:", module)

        llm = OllamaLLM(model="pnm-mistral:latest", base_url="http://ollama:11434")

        # Query ke Qdrant (sama dengan /ask)
        filter_query = {"module": module} if module else None
        all_docs = vectordb.similarity_search_with_score(question, k=15, filter=filter_query)

        threshold = 0.7
        filtered_docs = []
        for doc, score in all_docs:
            if score > threshold:
                continue
            if module and f"{module}.pdf" not in doc.metadata.get("source", ""):
                continue
            filtered_docs.append((doc, score))

        if not filtered_docs:
            return jsonify({
                "response": {
                    "jawaban": "Maaf, saya tidak menemukan jawaban untuk pertanyaan tersebut dalam modul-modul PNM.",
                    "sumber": []
                }
            })

        # Urutkan berdasarkan page_number
        filtered_docs.sort(key=lambda x: int(x[0].metadata.get('page_number', 0)))

        # Context + sumber
        context_parts = []
        sumber_set = set()
        for doc, score in filtered_docs:
            page = doc.metadata.get('page_number', '?')
            source = doc.metadata.get('source', 'unknown')
            context_parts.append(f"[Halaman {page}] {doc.page_content.strip()}")
            sumber_set.add(f"{source} (halaman {page})")

        context = "\n\n".join(context_parts)
        sumber_list = sorted(sumber_set, key=lambda s: int(s.split("halaman ")[-1].rstrip(")")))

        prompt = f"""
        Kamu adalah Sabrina, asisten AI ramah yang membantu menjawab pertanyaan berdasarkan dokumen resmi PNM.

        Pertanyaan user:
        {question}

        Dokumen relevan:
        {context}

        Jawaban:
        """

        def generate():
            try:
                for chunk in llm.stream(prompt):
                    text = getattr(chunk, "content", str(chunk))
                    if text:
                        yield text
                # setelah selesai, kirim meta sumber sekali di akhir
                yield "\n\n[SUMBER] " + json.dumps(sumber_list, ensure_ascii=False)
                yield "\n[DONE]"
            except Exception as e:
                yield f"[ERROR] {str(e)}"

        return Response(stream_with_context(generate()), mimetype="text/plain")

    except Exception as e:
        return jsonify({
            "error": "Terjadi kesalahan saat memproses permintaan.",
            "detail": str(e),
            "trace": traceback.format_exc()
        }), 500


if __name__ == "__main__":
    print("[SERVER] Aplikasi Flask berjalan di http://0.0.0.0:5000")
    app.run(debug=True, host="0.0.0.0", port=5000)
