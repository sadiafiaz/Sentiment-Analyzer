"""
Amazon Review Sentiment Analyzer - Flask Backend
--------------------------------------------------
Loads the trained deep learning model (saved as lstm_model.keras) and the
fitted Keras Tokenizer (tokenizer.pkl) exactly as they were produced in the
training notebook. No retraining, no re-fitting of the tokenizer happens here.

IMPORTANT (read this before you present the project):
The file is named `lstm_model.keras` for historical reasons, but the actual
architecture saved inside it is a GRU-based model (Embedding -> GRU -> GRU ->
Dense -> Dense/sigmoid), not a Bidirectional LSTM. This was verified by
loading the file and inspecting model.summary(). This app describes the
model accurately as "GRU" everywhere it matters (predictions, API, UI copy)
so nothing here overstates what was actually deployed.
"""

import os
import re
import pickle

import numpy as np
from flask import Flask, request, jsonify, render_template

import nltk
from nltk.corpus import stopwords
from nltk.stem import PorterStemmer

from tensorflow.keras.models import load_model 
model = load_model("GRU_model.keras")
from tensorflow.keras.preprocessing.sequence import pad_sequences

# ---------------------------------------------------------------------------
# Configuration (must match the notebook exactly)
# ---------------------------------------------------------------------------
MAX_WORDS = 1000
MAX_SEQUENCE_LENGTH = 100

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
MODEL_PATH = os.path.join(BASE_DIR, "GRU_model.keras")
TOKENIZER_PATH = os.path.join(BASE_DIR, "tokenizer.pkl")

# ---------------------------------------------------------------------------
# Flask app setup
# ---------------------------------------------------------------------------
app = Flask(__name__)

# ---------------------------------------------------------------------------
# NLTK setup (download stopwords locally if not already present)
# ---------------------------------------------------------------------------
try:
    stopwords.words("english")
except LookupError:
    nltk.download("stopwords")

STOP_WORDS = set(stopwords.words("english"))
STEMMER = PorterStemmer()

# ---------------------------------------------------------------------------
# Load model and tokenizer ONCE at startup (not per-request)
# ---------------------------------------------------------------------------
print("Loading model from:", MODEL_PATH)
model = load_model(MODEL_PATH)
print("Model loaded. Summary:")
model.summary()

print("Loading tokenizer from:", TOKENIZER_PATH)
with open(TOKENIZER_PATH, "rb") as f:
    tokenizer = pickle.load(f)
print("Tokenizer loaded. num_words =", tokenizer.num_words)


# ---------------------------------------------------------------------------
# Text cleaning - IDENTICAL to the notebook's clean_text() function
# ---------------------------------------------------------------------------
def clean_text(text: str) -> str:
    """Lowercase, strip non-alphabetic characters, remove stopwords, stem.

    This must stay byte-for-byte equivalent to the notebook's clean_text()
    function, or predictions will not match what the model was trained on.
    """
    text = text.lower()
    text = re.sub(r"[^a-z\s]", "", text)
    words = text.split()
    words = [STEMMER.stem(word) for word in words if word not in STOP_WORDS]
    return " ".join(words)


def predict_sentiment(raw_text: str):
    """Run the full pipeline: clean -> tokenize -> pad -> predict.

    Returns a dict with sentiment label, confidence (%), and raw score.
    """
    cleaned = clean_text(raw_text)

    sequence = tokenizer.texts_to_sequences([cleaned])
    padded = pad_sequences(sequence, maxlen=MAX_SEQUENCE_LENGTH)

    # Same decision rule as the notebook: prediction > 0.5 => Positive
    score = float(model.predict(padded, verbose=0)[0][0])
    is_positive = score > 0.5

    # Confidence reflects how sure the model is of the class it actually
    # picked. For a positive call, confidence = score. For a negative call,
    # confidence = 1 - score (distance from the "positive" end of the sigmoid).
    confidence = score if is_positive else (1.0 - score)

    return {
        "sentiment": "Positive" if is_positive else "Negative",
        "confidence": round(confidence * 100, 2),
        "score": round(score, 4),
        "cleaned_text": cleaned,
    }


# ---------------------------------------------------------------------------
# Routes
# ---------------------------------------------------------------------------
@app.route("/", methods=["GET", "POST"])
def index():
    """Simple server-rendered page: submit the form, get the result back
    in the same page (classic Flask/Jinja pattern, no JavaScript required).
    """
    if request.method == "POST":
        review_text = request.form.get("text", "").strip()

        if not review_text:
            return render_template("index.html", error="Please enter a review before analyzing.")

        result = predict_sentiment(review_text)
        return render_template(
            "index.html",
            review_text=review_text,
            sentiment=result["sentiment"],
            confidence=result["confidence"],
            score=result["score"],
        )

    return render_template("index.html")


@app.route("/predict", methods=["POST"])
def predict():
    data = request.get_json(silent=True)

    if not data or "text" not in data:
        return jsonify({"error": "Request must include a 'text' field."}), 400

    review_text = str(data["text"]).strip()

    if not review_text:
        return jsonify({"error": "Please enter a review before analyzing."}), 400

    if len(review_text) > 5000:
        return jsonify({"error": "Review text is too long (max 5000 characters)."}), 400

    try:
        result = predict_sentiment(review_text)
        return jsonify(result), 200
    except Exception as exc:  # pragma: no cover - defensive guard for the demo
        app.logger.exception("Prediction failed")
        return jsonify({"error": "Something went wrong while analyzing the review."}), 500


@app.route("/health")
def health():
    return jsonify({"status": "ok"}), 200


if __name__ == "__main__":
    port = int(os.environ.get("PORT", 5000))
    app.run(host="0.0.0.0", port=port)
