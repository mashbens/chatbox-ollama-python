from flask import Flask, request, jsonify, render_template_string
import time
from langchain_community.vectorstores import Chroma
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.llms import Ollama
from langchain.chains import RetrievalQA

app = Flask(__name__)

def get_qa_chain():
    embedding = HuggingFaceEmbeddings(
        model_name="BAAI/bge-m3",
        model_kwargs={"device": "cpu"},
        encode_kwargs={"normalize_embeddings": True}
    )
    vectordb = Chroma(persist_directory="db/", embedding_function=embedding)
    retriever = vectordb.as_retriever()
    llm = Ollama(model="pnm-mistral", base_url="http://ollama:11434")  # Pastikan sudah pull model
    qa = RetrievalQA.from_chain_type(llm=llm, retriever=retriever)
    return qa

qa_chain = get_qa_chain()
@app.route("/chat", methods=["POST"])
def chat():
    data = request.get_json()
    query = data.get("message", "")
    start = time.time()
    try:
        answer = qa_chain.run(query)
        duration = time.time() - start
        return jsonify({
            "answer": answer,
            "time": f"{duration:.2f} detik"
        })
    except Exception as e:
        return jsonify({"error": str(e)}), 500

if __name__ == "__main__":
    app.run(debug=True, host="0.0.0.0", port=5000)
