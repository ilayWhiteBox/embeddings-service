from flask import Flask, request, jsonify
from sentence_transformers import SentenceTransformer
import torch

app = Flask(__name__)

device = "cuda" if torch.cuda.is_available() else "cpu"
model = SentenceTransformer("all-mpnet-base-v2")
model.to(device)

@app.route("/embed", methods=["POST"])
def generate_embeddings():
    try:
        data = request.get_json(force=True)
        inputs = data.get("inputs", [])
        if not isinstance(inputs, list) or not all(isinstance(text, str) for text in inputs):
            return jsonify({"error": "Request must contain 'inputs': List[str]"}), 400

        embeddings = model.encode(inputs, convert_to_tensor=False)
        return jsonify({"embeddings": [emb.tolist() for emb in embeddings]})
    except Exception as e:
        return jsonify({"error": str(e)}), 500

@app.route("/healthz", methods=["GET"])
def health_check():
    return jsonify({"status": "ok"})
