"""Flask backend: serves the UI and a /api/predict endpoint."""
import json
import os

import joblib
from flask import Flask, jsonify, request, send_from_directory
from preprocessing import preprocess

BASE = os.path.dirname(os.path.abspath(__file__))
app = Flask(__name__, static_folder=os.path.join(BASE, "static"), static_url_path="/static")
model = joblib.load(os.path.join(BASE, "model.joblib"))


@app.get("/")
def index():
    return send_from_directory(app.static_folder, "index.html")


@app.get("/api/info")
def info():
    with open(os.path.join(BASE, "metrics.json")) as f:
        return jsonify(json.load(f))


@app.post("/api/predict")
def predict():
    data = request.get_json(silent=True) or {}
    try:
        x = preprocess(data.get("pixels"))
    except (ValueError, TypeError) as e:
        return jsonify(error=str(e)), 400
    probs = model.predict_proba(x)[0]
    return jsonify(prediction=int(probs.argmax()),
                   confidence=round(float(probs.max()), 4),
                   probabilities=[round(float(p), 4) for p in probs])

if __name__ == "__main__":
    app.run(host="127.0.0.1", port=5000, debug=False)