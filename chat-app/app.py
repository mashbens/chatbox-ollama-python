from flask import Flask, request, jsonify, Response, stream_with_context
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_ollama import OllamaLLM
from langchain_qdrant import Qdrant
from qdrant_client import QdrantClient
from functools import wraps
import traceback
import json
import requests
import base64

# --- Konfigurasi Flask ---
app = Flask(__name__)

USERNAME = "admin"
PASSWORD = "admin123"

def requires_auth(f):
    @wraps(f)
    def decorated(*args, **kwargs):
        auth_header = request.headers.get("Authorization", "")
        if not auth_header.startswith("Basic "):
            return Response(
                json.dumps({"error": "Unauthorized"}),
                401,
                {
                    "WWW-Authenticate": 'Basic realm="Login Required"',
                    "Content-Type": "application/json"
                }
            )

        try:
            # Decode token base64
            encoded = auth_header.split(" ", 1)[1]
            decoded = base64.b64decode(encoded).decode("utf-8")
            username, password = decoded.split(":", 1)
        except Exception:
            return Response(
                json.dumps({"error": "Invalid Authorization header"}),
                401,
                {"Content-Type": "application/json"}
            )

        if username != USERNAME or password != PASSWORD:
            return Response(
                json.dumps({"error": "Unauthorized"}),
                401,
                {
                    "WWW-Authenticate": 'Basic realm="Login Required"',
                    "Content-Type": "application/json"
                }
            )

        # kalau lolos → lanjut ke endpoint
        return f(*args, **kwargs)
    return decorated


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

# --- Endpoint Chat Normal (tetap persis) ---
@app.route("/ask", methods=["POST"])
@requires_auth
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

        prompt = f"""
        Kamu adalah Sabrina, asisten AI ramah yang membantu menjawab pertanyaan berdasarkan dokumen resmi PNM.

        Tugasmu:
        1. Jawab pertanyaan user dengan jelas, ringkas, dan terstruktur.
        2. Jika informasi ada di dokumen → gunakan dokumen.
        3. Jika tidak ada → beri jawaban umum yang relevan, lalu sarankan langkah praktis.
        4. Setelah memberi jawaban, tambahkan satu pertanyaan lanjutan atau tawaran ide.
        - Letakkan follow-up di baris baru setelah jawaban utama pisahkan dengan satu enter.
        - Follow-up harus bervariasi (tidak selalu "Mau saya...", bisa juga "Apakah kamu ingin…", "Kalau tertarik saya bisa…", dll).

        Dokumen relevan:
        {context}

        Pertanyaan user:
        {question}

        Jawaban:
        """

        print("[INFO] Mengirim prompt ke LLM...")
        jawaban = llm.invoke(prompt)
        print("[INFO] Jawaban LLM diterima.")

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


@app.route("/ask-stream", methods=["POST"])
@requires_auth
def chat_stream():
    data = request.get_json()
    question = data.get("question", "").strip()
    module = data.get("module", "").strip()

    if not question:
        return jsonify({"error": "Parameter 'question' tidak boleh kosong."}), 400

    def generate():
        try:
            print("========== NEW STREAM REQUEST ==========")
            print("Pertanyaan:", question)
            print("Module filter:", module)

            # --- Step 1: Query Qdrant persis kayak /ask ---
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
                # kalau gak ada dokumen relevan → kirim error langsung
                yield json.dumps({
                    "response": {
                        "jawaban": "Maaf, saya tidak menemukan jawaban untuk pertanyaan tersebut dalam modul-modul PNM yang diberikan.",
                        "sumber": []
                    }
                }) + "\n"
                return

            # Urutkan berdasarkan halaman
            filtered_docs.sort(key=lambda x: int(x[0].metadata.get('page_number', 0)))

            # Build context & sumber
            context_parts = []
            sumber_set = set()
            for doc, score in filtered_docs:
                page = doc.metadata.get('page_number', '?')
                source = doc.metadata.get('source', 'unknown')
                context_parts.append(f"[Halaman {page}] {doc.page_content.strip()}")
                sumber_set.add(f"{source} (halaman {page})")

            context = "\n\n".join(context_parts)
            sumber_list = sorted(sumber_set, key=lambda s: int(s.split("halaman ")[-1].rstrip(")")))

            # --- Step 2: Bangun prompt sama persis kayak /ask ---
            prompt = f"""
            Kamu adalah Sabrina, asisten AI ramah yang membantu menjawab pertanyaan berdasarkan dokumen resmi PNM.

            Tugasmu:
            1. Jawab pertanyaan user dengan jelas, ringkas, dan terstruktur.
            2. Jika informasi ada di dokumen → gunakan dokumen.
            3. Jika tidak ada → beri jawaban umum yang relevan, lalu sarankan langkah praktis.
            4. Setelah memberi jawaban, tambahkan satu pertanyaan lanjutan atau tawaran ide.
            - Letakkan follow-up di baris baru setelah jawaban utama pisahkan dengan satu enter.
            - Follow-up harus bervariasi (tidak selalu "Mau saya...", bisa juga "Apakah kamu ingin…", "Kalau tertarik saya bisa…", dll).

            Dokumen relevan:
            {context}

            Pertanyaan user:
            {question}

            Jawaban:
            """

            # --- Step 3: Panggil Ollama API dengan stream=True ---
            ollama_url = "http://ollama:11434/api/generate"
            payload = {
                "model": "pnm-mistral:latest",
                "prompt": prompt,
                "stream": True
            }

            with requests.post(ollama_url, json=payload, stream=True, timeout=None) as response:
                for line in response.iter_lines():
                    if line:
                        decoded = line.decode("utf-8") if isinstance(line, bytes) else line
                        yield decoded + "\n"

                # --- Step 4: Tambahin sumber setelah selesai ---
                final_data = {
                    "sumber": sumber_list
                }
                yield json.dumps(final_data) + "\n"

        except Exception as e:
            error_data = {"error": str(e)}
            yield json.dumps(error_data) + "\n"

    return Response(
        stream_with_context(generate()),
        content_type='application/json',
        headers={
            'Cache-Control': 'no-cache',
            'Connection': 'keep-alive',
            'X-Accel-Buffering': 'no'
        }
    )

# @app.route("/ask-stream-simple", methods=["POST"])
# def chat_stream_simple():
#     data = request.get_json()
#     question = data.get("question", "").strip()
    
#     if not question:
#         return jsonify({"error": "Parameter 'question' tidak boleh kosong."}), 400

#     def generate():
#         try:
#             ollama_url = "http://ollama:11434/api/generate"
#             payload = {
#                 "model": "pnm-mistral:latest",
#                 "prompt": question,
#                 "stream": True
#             }
            
#             # timeout=None biar gak dipotong
#             with requests.post(ollama_url, json=payload, stream=True, timeout=None) as response:
#                 for line in response.iter_lines():
#                     if line:
#                         # pastikan decode ke string
#                         decoded = line.decode("utf-8") if isinstance(line, bytes) else line
#                         yield decoded + "\n"

#         except Exception as e:
#             error_data = {"error": str(e)}
#             yield json.dumps(error_data) + "\n"

#     return Response(
#         stream_with_context(generate()),
#         content_type='application/json',
#         headers={
#             'Cache-Control': 'no-cache',
#             'Connection': 'keep-alive',
#             'X-Accel-Buffering': 'no'
#         }
#     )



if __name__ == "__main__":
    print("[SERVER] Aplikasi Flask berjalan di http://0.0.0.0:5000")
    app.run(debug=True, host="0.0.0.0", port=5000)