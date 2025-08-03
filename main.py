from flask import Flask, request, jsonify
import logging
import numpy as np
import torch
from sentence_transformers import SentenceTransformer
from transformers import pipeline
from sklearn.cluster import AgglomerativeClustering
from sklearn.metrics.pairwise import cosine_similarity

# Initialize Flask app
app = Flask(__name__)

# Configure logging
logging.basicConfig(level=logging.INFO)

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

@app.route("/cluster", methods=["POST"])
def cluster_embeddings():
    try:
        body = request.get_json(force=True)

        embeddings = np.array(body.get("embeddings", []))
        distance_threshold = float(body.get("distance_threshold", 0.6))

        if embeddings.ndim == 1:
            embeddings = embeddings.reshape(1, -1)
        if embeddings.shape[0] < 2:
            return jsonify({"labels": list(range(embeddings.shape[0]))})

        # Use 1 - cosine similarity to preserve original logic
        similarity_matrix = cosine_similarity(embeddings)
        distance_matrix = 1 - similarity_matrix

        clustering = AgglomerativeClustering(
            n_clusters=None,
            metric="precomputed",
            linkage="average",
            distance_threshold=distance_threshold
        )
        labels = clustering.fit_predict(distance_matrix)

        return jsonify({"labels": labels.tolist()})
    except Exception as e:
        logging.exception("Error occurred in /cluster endpoint")
        return jsonify({"error": str(e)}), 500

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
        logging.exception("Error occurred in /embed endpoint")
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
        logging.exception("Error occurred in /sentiment endpoint")
        return jsonify({"error": str(e)}), 500

@app.route("/healthz", methods=["GET"])
def health_check():
    return jsonify({"status": "ok"})

