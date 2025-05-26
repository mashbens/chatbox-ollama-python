from flask import Flask, request, jsonify
from langchain.chains import RetrievalQA
from langchain_community.vectorstores import Chroma
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_ollama import OllamaLLM
import traceback  # Tambahkan ini untuk membantu debugging

app = Flask(__name__)

def load_vectordb():
    embedding = HuggingFaceEmbeddings(
        model_name="BAAI/bge-m3",
        model_kwargs={"device": "cpu"},
        encode_kwargs={"normalize_embeddings": True}
    )
    vectordb = Chroma(persist_directory="db", embedding_function=embedding)
    return vectordb

@app.route("/ask", methods=["POST"])
def chat():
    data = request.get_json()
    question = data.get("question", "").strip()
    module = data.get("module", "").strip()

    if not question:
        return jsonify({"error": "Parameter 'question' tidak boleh kosong."}), 400

    try:
        print("Pertanyaan:", question)
        print("Module filter:", module)

        # Load vector DB & model
        vectordb = load_vectordb()
        llm = OllamaLLM(model="pnm-mistral", base_url="http://ollama:11434")

        # Set filter jika ada modul
        retriever_kwargs = {"k": 5}
        if module:
            retriever_kwargs["filter"] = {"source": f"{module}.pdf"}

        retriever = vectordb.as_retriever(search_kwargs=retriever_kwargs)

        # QA chain
        qa_chain = RetrievalQA.from_chain_type(
            llm=llm,
            retriever=retriever,
            return_source_documents=True
        )

        result = qa_chain.invoke({"query": question})
        jawaban = result["result"]
        sumber_docs = result.get("source_documents", [])

        sumber_list = list({doc.metadata.get("source", "unknown") for doc in sumber_docs})

        return jsonify({
            "response": {
                "jawaban": jawaban.strip(),
                "sumber": sumber_list if sumber_list else ["general"]
            }
        })

    except Exception as e:
        error_message = str(e)
        traceback_str = traceback.format_exc()
        print("Terjadi error:", error_message)
        print(traceback_str)  # Cetak stack trace ke console/log

        return jsonify({
            "error": "Terjadi kesalahan saat memproses permintaan.",
            "detail": error_message,
            "trace": traceback_str  # Hati-hati expose ini di production!
        }), 500

if __name__ == "__main__":
    app.run(debug=True, host="0.0.0.0", port=5000)