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

# --- Endpoint Chat Normal (tetap persis) ---
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
        - Letakkan follow-up di baris baru setelah jawaban utama (pisahkan dengan satu enter).
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

# --- Endpoint Chat Streaming (versi baru) ---
@app.route("/ask-stream", methods=["POST"])
def chat_stream():
    data = request.get_json()
    question = data.get("question", "").strip()
    module = data.get("module", "").strip()

    if not question:
        return jsonify({"error": "Parameter 'question' tidak boleh kosong."}), 400

    def generate():
        try:
            print("========== NEW STREAMING REQUEST ==========")
            print("Pertanyaan:", question)
            print("Module filter:", module)

            # Query ke Qdrant (sama seperti endpoint normal)
            print("[INFO] Mencari dokumen relevan dari Qdrant...")
            filter_query = {"module": module} if module else None
            all_docs = vectordb.similarity_search_with_score(question, k=15, filter=filter_query)

            # Filter dan simpan (doc, score)
            threshold = 0.7
            filtered_docs = []
            for doc, score in all_docs:
                if score > threshold:
                    continue
                if module and f"{module}.pdf" not in doc.metadata.get("source", ""):
                    continue
                filtered_docs.append((doc, score))

            # Urutkan berdasarkan page_number
            filtered_docs.sort(key=lambda x: int(x[0].metadata.get('page_number', 0)))

            # Susun context dan sumber
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

            Tugasmu:
            1. Jawab pertanyaan user dengan jelas, ringkas, dan terstruktur.
            2. Jika informasi ada di dokumen → gunakan dokumen.
            3. Jika tidak ada → beri jawaban umum yang relevan, lalu sarankan langkah praktis.
            4. Setelah memberi jawaban, tambahkan satu pertanyaan lanjutan atau tawaran ide.
            - Letakkan follow-up di baris baru setelah jawaban utama (pisahkan dengan satu enter).
            - Follow-up harus bervariasi (tidak selalu "Mau saya...", bisa juga "Apakah kamu ingin…", "Kalau tertarik saya bisa…", dll).

            Dokumen relevan:
            {context}

            Pertanyaan user:
            {question}

            Jawaban:
            """

            print("[INFO] Mengirim prompt ke LLM dengan streaming...")
            
            # Kirim request streaming langsung ke Ollama
            import requests
            
            ollama_url = "http://ollama:11434/api/generate"
            payload = {
                "model": "pnm-mistral:latest",
                "prompt": prompt,
                "stream": True
            }
            
            response = requests.post(ollama_url, json=payload, stream=True)
            
            # Stream response dari Ollama ke client
            for line in response.iter_lines():
                if line:
                    line_str = line.decode('utf-8')
                    try:
                        data = json.loads(line_str)
                        
                        # Format response sesuai dengan contoh curl
                        response_data = {
                            "model": data.get("model", "pnm-mistral:latest"),
                            "created_at": data.get("created_at", ""),
                            "response": data.get("response", ""),
                            "done": data.get("done", False)
                        }
                        
                        # Kirim setiap chunk response
                        yield f"data: {json.dumps(response_data)}\n\n"
                        
                        # Jika selesai, kirim sumber dokumen
                        if data.get("done", False):
                            final_data = {
                                "model": "pnm-mistral:latest",
                                "created_at": data.get("created_at", ""),
                                "response": "",
                                "done": True,
                                "sumber": sumber_list
                            }
                            yield f"data: {json.dumps(final_data)}\n\n"
                            break
                            
                    except json.JSONDecodeError:
                        continue

        except Exception as e:
            error_message = str(e)
            traceback_str = traceback.format_exc()
            print("Terjadi error:", error_message)
            print(traceback_str)
            
            error_data = {
                "error": "Terjadi kesalahan saat memproses permintaan.",
                "detail": error_message
            }
            yield f"data: {json.dumps(error_data)}\n\n"

    return Response(stream_with_context(generate()), content_type='text/plain')

if __name__ == "__main__":
    print("[SERVER] Aplikasi Flask berjalan di http://0.0.0.0:5000")
    app.run(debug=True, host="0.0.0.0", port=5000)