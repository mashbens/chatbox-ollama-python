from flask import Flask, request, jsonify
from langchain.chains import RetrievalQA
from langchain_community.vectorstores import Chroma
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.llms import Ollama

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
        vectordb = load_vectordb()
        llm = Ollama(model="pnm-mistral", base_url="http://localhost:11434")

        retriever_kwargs = {"k": 5}
        if module:
            retriever_kwargs["filter"] = {"source": f"{module}.pdf"}

        retriever = vectordb.as_retriever(search_kwargs=retriever_kwargs)

        qa_chain = RetrievalQA.from_chain_type(
            llm=llm,
            retriever=retriever,
            return_source_documents=True
        )

        result = qa_chain.invoke({"query": question})
        jawaban = result["result"]

        return jsonify({
            "response": {
                "jawaban": jawaban.strip(),
                "sumber": module if module else "general"
            }
        })

    except Exception as e:
        return jsonify({"error": str(e)}), 500

if __name__ == "__main__":
    app.run(debug=True, host="0.0.0.0", port=31133)
