"""
Toxic Comment Classifier - FastAPI Microservice
Real-time toxicity detection: single comment + batch, multi-label output
"""

import os
import torch
import numpy as np
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field
from typing import List, Optional, Dict
import uvicorn
import logging

from model import TextCNN, ToxicDataset
from preprocessing import TextPreprocessor, LABEL_COLUMNS
from torch.utils.data import DataLoader

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

app = FastAPI(
    title="Toxic Comment Classifier API",
    description="Real-time multi-label toxicity detection using TextCNN",
    version="1.0.0",
)
app.add_middleware(CORSMiddleware, allow_origins=["*"], allow_methods=["*"], allow_headers=["*"])

# ─── Global State ─────────────────────────────────────────────────────────────
MODEL: Optional[TextCNN] = None
PREPROCESSOR: Optional[TextPreprocessor] = None
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
THRESHOLD = float(os.getenv("THRESHOLD", "0.5"))


def load_artifacts():
    global MODEL, PREPROCESSOR
    artifacts_dir = os.getenv("ARTIFACTS_DIR", "artifacts")
    model_path = os.path.join(artifacts_dir, "textcnn_model.pt")

    PREPROCESSOR = TextPreprocessor().load(artifacts_dir)

    checkpoint = torch.load(model_path, map_location=DEVICE)
    MODEL = TextCNN(
        vocab_size=checkpoint["vocab_size"],
        embed_dim=checkpoint["embed_dim"],
        num_filters=checkpoint["num_filters"],
        filter_sizes=checkpoint["filter_sizes"],
        num_classes=checkpoint["num_classes"],
        dropout=checkpoint["dropout"],
    ).to(DEVICE)
    MODEL.load_state_dict(checkpoint["model_state_dict"])
    MODEL.eval()
    logger.info(f"Model loaded | device={DEVICE} | vocab={PREPROCESSOR.vocab_size}")


@app.on_event("startup")
async def startup():
    load_artifacts()


# ─── Schemas ──────────────────────────────────────────────────────────────────
class CommentRequest(BaseModel):
    text: str = Field(..., min_length=1, max_length=5000,
                      example="You are absolutely terrible and should be ashamed.")
    threshold: float = Field(THRESHOLD, ge=0.0, le=1.0)


class ToxicityResult(BaseModel):
    text: str
    is_toxic: bool
    overall_toxicity_score: float
    labels: Dict[str, float]
    flagged_categories: List[str]
    severity: str  # "CLEAN" | "MILD" | "MODERATE" | "SEVERE"


class BatchRequest(BaseModel):
    texts: List[str] = Field(..., min_items=1, max_items=512)
    threshold: float = Field(THRESHOLD, ge=0.0, le=1.0)


class BatchResponse(BaseModel):
    results: List[ToxicityResult]
    total: int
    toxic_count: int
    clean_count: int


# ─── Helpers ──────────────────────────────────────────────────────────────────
def severity_label(score: float, n_flags: int) -> str:
    if n_flags == 0 or score < 0.3:
        return "CLEAN"
    elif score < 0.5 or n_flags == 1:
        return "MILD"
    elif score < 0.7 or n_flags <= 2:
        return "MODERATE"
    return "SEVERE"


@torch.no_grad()
def run_inference(texts: List[str]) -> np.ndarray:
    """Return sigmoid probabilities (N, num_classes)."""
    X = PREPROCESSOR.batch_encode(texts)
    dataset = ToxicDataset(X)
    loader = DataLoader(dataset, batch_size=256)
    all_probs = []
    for X_batch in loader:
        X_batch = X_batch.to(DEVICE)
        logits = MODEL(X_batch)
        all_probs.append(torch.sigmoid(logits).cpu().numpy())
    return np.vstack(all_probs)


def build_result(text: str, probs: np.ndarray, threshold: float) -> ToxicityResult:
    label_probs = {label: round(float(prob), 4) for label, prob in zip(LABEL_COLUMNS, probs)}
    flagged = [label for label, prob in label_probs.items() if prob >= threshold]
    overall_score = float(probs.max())
    return ToxicityResult(
        text=text[:200] + "..." if len(text) > 200 else text,
        is_toxic=len(flagged) > 0,
        overall_toxicity_score=round(overall_score, 4),
        labels=label_probs,
        flagged_categories=flagged,
        severity=severity_label(overall_score, len(flagged)),
    )


# ─── Endpoints ────────────────────────────────────────────────────────────────
@app.get("/health")
async def health():
    return {
        "status": "ok",
        "model_loaded": MODEL is not None,
        "device": str(DEVICE),
        "vocab_size": PREPROCESSOR.vocab_size if PREPROCESSOR else None,
        "labels": LABEL_COLUMNS,
        "threshold": THRESHOLD,
    }


@app.post("/classify", response_model=ToxicityResult)
async def classify(request: CommentRequest):
    """Classify a single comment for toxicity."""
    if MODEL is None:
        raise HTTPException(503, "Model not loaded")
    try:
        probs = run_inference([request.text])[0]
        return build_result(request.text, probs, request.threshold)
    except Exception as e:
        logger.error(f"Classification error: {e}")
        raise HTTPException(500, str(e))


@app.post("/classify/batch", response_model=BatchResponse)
async def classify_batch(request: BatchRequest):
    """Classify a batch of comments."""
    if MODEL is None:
        raise HTTPException(503, "Model not loaded")
    try:
        all_probs = run_inference(request.texts)
        results = [
            build_result(text, probs, request.threshold)
            for text, probs in zip(request.texts, all_probs)
        ]
        toxic_count = sum(1 for r in results if r.is_toxic)
        return BatchResponse(
            results=results,
            total=len(results),
            toxic_count=toxic_count,
            clean_count=len(results) - toxic_count,
        )
    except Exception as e:
        raise HTTPException(500, str(e))


@app.get("/model/info")
async def model_info():
    if MODEL is None:
        raise HTTPException(503, "Model not loaded")
    return {
        "architecture": "TextCNN",
        "vocab_size": PREPROCESSOR.vocab_size,
        "max_len": PREPROCESSOR.max_len,
        "labels": LABEL_COLUMNS,
        "total_parameters": sum(p.numel() for p in MODEL.parameters()),
        "device": str(DEVICE),
        "threshold": THRESHOLD,
    }


if __name__ == "__main__":
    uvicorn.run("main:app", host="0.0.0.0", port=8003, reload=False)
