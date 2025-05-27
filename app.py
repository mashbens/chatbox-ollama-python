from flask import Flask, request, jsonify
from langchain_community.vectorstores import Chroma
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_ollama import OllamaLLM
import traceback

app = Flask(__name__)

def load_vectordb():
    print("[INFO] Memuat vector database...")
    embedding = HuggingFaceEmbeddings(
        model_name="BAAI/bge-m3",
        model_kwargs={"device": "cpu"},
        encode_kwargs={"normalize_embeddings": True}
    )
    vectordb = Chroma(persist_directory="db", embedding_function=embedding)
    print("[INFO] Vector DB berhasil dimuat.")
    return vectordb

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

        # Load vector DB & model
        vectordb = load_vectordb()
        llm = OllamaLLM(model="pnm-mistral", base_url="http://localhost:11434")

        # Ambil dokumen berdasarkan similarity + score
        print("[INFO] Mencari dokumen relevan dari vector DB...")
        all_docs = vectordb.similarity_search_with_score(question, k=5)

        print("[DEBUG] Dokumen hasil similarity_search_with_score:")
        for i, (doc, score) in enumerate(all_docs):
            print(f"  {i+1}. Score: {score:.4f} | Source: {doc.metadata.get('source', 'unknown')}")

        # Filter berdasarkan threshold dan modul (jika ada)
        threshold = 1.1
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

        # Gabungkan isi dokumen
        context = "\n\n".join([doc.page_content for doc in filtered_docs])
        sumber_list = list({doc.metadata.get("source", "unknown") for doc in filtered_docs})

        print("[DEBUG] Daftar sumber dokumen:", sumber_list)

        # Buat prompt
        prompt = f"""
Kamu adalah AI PNM. Jawablah pertanyaan berikut hanya berdasarkan isi modul. 
Jika jawabannya tidak ditemukan dalam modul, katakan:
"Maaf, saya tidak menemukan jawaban untuk pertanyaan tersebut dalam modul-modul PNM yang diberikan."

Pertanyaan: {question}

Isi modul:
{context}

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

if __name__ == "__main__":
    print("[SERVER] Aplikasi Flask berjalan di http://0.0.0.0:5000")
    app.run(debug=True, host="0.0.0.0", port=5000)
