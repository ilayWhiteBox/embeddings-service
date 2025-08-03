from flask import Flask, request, jsonify
from sentence_transformers import SentenceTransformer
from transformers import pipeline
import torch

app = Flask(__name__)

# Detect device
device = 0 if torch.cuda.is_available() else -1

# Load sentence embedding model
embedding_model = SentenceTransformer("all-mpnet-base-v2")
embedding_model.to("cuda" if device == 0 else "cpu")

# Load sentiment model pipeline
sentiment_pipeline = pipeline(
    "sentiment-analysis",
    model="cardiffnlp/twitter-roberta-base-sentiment-latest",
    tokenizer="cardiffnlp/twitter-roberta-base-sentiment-latest",
    device=device
)

@app.route("/embed", methods=["POST"])
def generate_embeddings():
    try:
        data = request.get_json(force=True)
        inputs = data.get("inputs", [])
        if not isinstance(inputs, list) or not all(isinstance(text, str) for text in inputs):
            return jsonify({"error": "Request must contain 'inputs': List[str]"}), 400

        embeddings = embedding_model.encode(inputs, convert_to_tensor=False)
        return jsonify({"embeddings": [emb.tolist() for emb in embeddings]})
    except Exception as e:
        return jsonify({"error": str(e)}), 500

@app.route("/sentiment", methods=["POST"])
def analyze_sentiment():
    try:
        data = request.get_json(force=True)
        inputs = data.get("inputs", [])
        if not isinstance(inputs, list) or not all(isinstance(text, str) for text in inputs):
            return jsonify({"error": "Request must contain 'inputs': List[str]"}), 400

        results = sentiment_pipeline(inputs)
        return jsonify({"results": results})
    except Exception as e:
        return jsonify({"error": str(e)}), 500

@app.route("/healthz", methods=["GET"])
def health_check():
    return jsonify({"status": "ok"})
