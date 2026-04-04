# Toxic Comment Classifier

Multi-label toxicity detection using TextCNN (Kim 2014) with a complete NLP preprocessing pipeline and FastAPI microservice for real-time inference.

## Architecture
```
Raw Text → TextPreprocessor (clean → tokenize → encode → pad)
                ↓
          TextCNN Model
    Embedding → Conv1d(k=2,3,4,5) → ReLU → MaxPool
              → Concat → Dropout → FC(6)
                ↓
     Sigmoid → Multi-label probabilities
    [toxic, severe_toxic, obscene, threat, insult, identity_hate]
                ↓
         FastAPI microservice
     POST /classify  |  POST /classify/batch
```

## Project Structure
```
toxic_comment/
├── preprocessing.py   # TextPreprocessor: clean, vocab, encode, pad
├── model.py           # TextCNN architecture + ToxicDataset
├── train.py           # Training loop, MLflow tracking, early stopping
├── main.py            # FastAPI service: /classify, /classify/batch
├── Dockerfile
└── requirements.txt
```

## Quick Start

### 1. Install
```bash
pip install -r requirements.txt
```

### 2. Get Data
Download the [Jigsaw Toxic Comment Classification dataset](https://www.kaggle.com/c/jigsaw-toxic-comment-classification-challenge) and place at:
```
data/train.csv
```

### 3. Train
```bash
python train.py
```
Artifacts saved to `artifacts/` — model + preprocessor.
MLflow UI: `mlflow ui`

### 4. Serve
```bash
uvicorn main:app --host 0.0.0.0 --port 8003
```

### 5. Docker
```bash
docker build -t toxic-api .
docker run -p 8003:8003 toxic-api
```

## API Endpoints

| Method | Endpoint | Description |
|--------|----------|-------------|
| GET | /health | Health + model info |
| POST | /classify | Single comment |
| POST | /classify/batch | Up to 512 comments |
| GET | /model/info | Architecture details |

### Single Comment
```bash
curl -X POST http://localhost:8003/classify \
  -H "Content-Type: application/json" \
  -d '{"text": "You are absolutely terrible and should disappear.", "threshold": 0.5}'
```

### Response
```json
{
  "text": "You are absolutely terrible and should disappear.",
  "is_toxic": true,
  "overall_toxicity_score": 0.8341,
  "labels": {
    "toxic": 0.8341,
    "severe_toxic": 0.1023,
    "obscene": 0.2145,
    "threat": 0.0512,
    "insult": 0.6782,
    "identity_hate": 0.0341
  },
  "flagged_categories": ["toxic", "insult"],
  "severity": "SEVERE"
}
```

### Batch
```bash
curl -X POST http://localhost:8003/classify/batch \
  -H "Content-Type: application/json" \
  -d '{"texts": ["Hello world", "I hate you", "Great work!"], "threshold": 0.5}'
```

## Model Details
- **Architecture**: TextCNN with 4 filter sizes (2, 3, 4, 5), 128 filters each
- **Embedding**: 128-dim trainable embeddings
- **Loss**: BCEWithLogitsLoss with per-class positive weighting for imbalance
- **Optimizer**: Adam + CosineAnnealingLR
- **Labels**: 6 (toxic, severe_toxic, obscene, threat, insult, identity_hate)
- **Threshold**: Configurable via env var `THRESHOLD` (default 0.5)
