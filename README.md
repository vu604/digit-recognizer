# Digit Recognizer: a small full-stack ML project

Draw a digit in the browser, get a prediction and per-class probabilities back from a Python API.

```
Browser canvas ──(56x56 grayscale JSON)──▶ Flask /api/predict ──▶ preprocess ──▶ MLP ──▶ probabilities
```

| Layer | File | What it does |
|---|---|---|
| Model | `train.py` | Trains a shift-augmented RBF support vector classifier on scikit-learn's real 8x8 handwritten digits dataset. Saves `model.joblib` and the measured held-out accuracy in `metrics.json`. |
| Backend | `app.py` | Flask app. `POST /api/predict`, `GET /api/info`, and serves the UI. Shared preprocessing in `preprocessing.py` crops, centres and resizes the drawing to 8x8. |
| Frontend | `static/index.html` | Canvas drawing pad (mouse/touch), probability bars, light/dark theme. |
| Tests | `test_api.py` | Sends simulated drawings through the API and checks error handling. |

## Run it
```bash
pip install -r requirements.txt
python train.py        # ~1 min, writes model.joblib
python test_api.py     # optional sanity check
python app.py          # open http://127.0.0.1:5000
```
If you retrain while the Flask server is running, restart `app.py` and refresh the browser so it loads the new model and status.

## API
`POST /api/predict` with `{"pixels": [[0..255, ...], ...]}` (square 2D array, ink = high values)
returns `{"prediction": 7, "confidence": 0.98, "probabilities": [10 floats]}`.

## Limits and ideas
- The 8x8 handwritten digits dataset is small and low-resolution. `metrics.json` reports a held-out score, but unusual handwriting can still be misclassified.
- Next step: swap in MNIST (28x28) with a small CNN in PyTorch. Only `train.py` and `preprocess()` need to change.
- Add a "this was wrong" button that stores corrected samples for retraining.
