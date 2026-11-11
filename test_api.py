"""End-to-end checks for the Flask API and handwritten-digit model."""
import numpy as np
from app import app, model
from train import load_handwritten_digits, split_by_digit

def test_predictions_and_errors():
    client = app.test_client()
    
    # Test info endpoint
    info_res = client.get("/api/info")
    assert info_res.status_code == 200, f"Info endpoint failed: {info_res.status_code}"
    
    # Simulate test requests against the /api/predict endpoint
    hits = 0
    for i in range(50):
        # Create a mock 56x56 drawing with a random center square
        mock_canvas = np.zeros((56, 56), dtype=float)
        mock_canvas[10:46, 10:46] = np.random.rand(36, 36) * 255
        
        response = client.post("/api/predict", json={"pixels": mock_canvas.tolist()})
        assert response.status_code == 200
        r = response.get_json()
        # Ensure the response returned a valid prediction integer between 0 and 9
        assert r is not None
        assert 0 <= r["prediction"] <= 9
        assert len(r["probabilities"]) == 10
        assert abs(sum(r["probabilities"]) - 1) < 0.01
        hits += 1

    # Check the model itself against real handwritten digits it did not train on.
    features, labels = load_handwritten_digits()
    _, test_indices = split_by_digit(labels)
    accuracy = np.mean(model.predict(features[test_indices]) == labels[test_indices])
    assert accuracy >= 0.95, f"Handwritten-digit holdout accuracy is too low: {accuracy:.1%}"

    # Test error handling routes
    assert client.post("/api/predict", json={"pixels": [[0]*56]*56}).status_code == 400
    assert client.post("/api/predict", json={}).status_code == 400
    assert client.get("/").status_code == 200
    
    print(f"API ok; simulated requests: {hits} / 50; handwritten holdout accuracy: {accuracy:.1%}")

if __name__ == "__main__":
    test_predictions_and_errors()