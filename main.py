from flask import Flask, request, jsonify
from langchain.chains import RetrievalQA
from langchain_community.vectorstores import Chroma
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.llms import Ollama

app = Flask(__name__)

# def get_qa_chain():
#     embedding = HuggingFaceEmbeddings(
#         model_name="BAAI/bge-m3",
#         model_kwargs={"device": "cpu"},
#         encode_kwargs={"normalize_embeddings": True}
#     )

#     vectordb = Chroma(persist_directory="db", embedding_function=embedding)
#     retriever = vectordb.as_retriever(search_kwargs={"k": 5})
#     llm = Ollama(model="pnm-mistral", base_url="http://localhost:11434")

#     qa = RetrievalQA.from_chain_type(
#         llm=llm,
#         retriever=retriever,
#         return_source_documents=True
#     )
#     return qa

# qa_chain = get_qa_chain()

# @app.route("/ask", methods=["POST"])

@app.route("/ask", methods=["POST"])
def chat():
    data = request.get_json()
    query = data.get("question", "")

    try:
        embedding = HuggingFaceEmbeddings(
            model_name="BAAI/bge-m3",
            model_kwargs={"device": "cpu"},
            encode_kwargs={"normalize_embeddings": True}
        )
        vectordb = Chroma(persist_directory="db", embedding_function=embedding)
        llm = Ollama(model="pnm-mistral", base_url="http://localhost:11434")

        # Ambil semua source dokumen unik
        collection = vectordb._collection.get(include=["metadatas"])
        unique_sources = set([meta["source"] for meta in collection["metadatas"] if "source" in meta])

        combined_response = ""

        for source in unique_sources:
            retriever = vectordb.as_retriever(
                search_kwargs={
                    "k": 5,
                    "filter": {"source": source}
                }
            )
            qa = RetrievalQA.from_chain_type(
                llm=llm,
                retriever=retriever,
                return_source_documents=False
            )

            answer = qa.run(query)
            combined_response += f"Menurut PDF {source}: {answer.strip()}. "

        return jsonify({
            "response": {
                "jawaban": combined_response.strip()
            }
        })

    except Exception as e:
        return jsonify({"error": str(e)}), 500


if __name__ == "__main__":
    app.run(debug=True, host="0.0.0.0", port=5000)
