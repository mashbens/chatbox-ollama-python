from flask import Flask, request, jsonify
import os
from werkzeug.utils import secure_filename
from ingest import ingest_pdfs

app = Flask(__name__)

ALLOWED_EXTENSIONS = {'pdf'}

# BASE_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
# UPLOAD_FOLDER = os.path.join(BASE_DIR, "uploads")

UPLOAD_FOLDER = "/app/uploads"

app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER
os.makedirs(UPLOAD_FOLDER, exist_ok=True)


def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS


@app.route("/upload", methods=["POST"])
def upload_file():
    if 'file' not in request.files:
        return jsonify({"error": "No file uploaded"}), 400

    file = request.files['file']
    module = request.form.get('module')
    category = request.form.get('category', 'other')

    if not file or file.filename == '':
        return jsonify({"error": "No file selected"}), 400

    if not allowed_file(file.filename):
        return jsonify({"error": "Invalid file type"}), 400

    if not module:
        return jsonify({"error": "Module name is required"}), 400

    filename = secure_filename(file.filename)
    filepath = os.path.join(app.config['UPLOAD_FOLDER'], filename)
    file.save(filepath)

    print(f"✅ File disimpan di: {filepath}")

    file_info = [{
        "path": filepath,
        "module": module,
        "category": category
    }]

    ingest_pdfs(file_info)

    return jsonify({"message": "File uploaded and ingested successfully!"})


if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000)
