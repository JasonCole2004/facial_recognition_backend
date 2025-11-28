import io
import json
import os
from typing import List, Optional

import boto3
import cv2
import numpy as np
from deepface import DeepFace
from fastapi import FastAPI, File, UploadFile, HTTPException
from fastapi.responses import JSONResponse

# ---------- CONFIG ----------
AWS_REGION = "eu-west-2"  # AWS region
S3_BUCKET = "actor-face-database"  # bucket name here
EMBEDDINGS_KEY = "embeddings.json"   # path to embeddings in S3

MODEL_NAME = "Facenet"
SIM_THRESHOLD = 0.7
# -----------------------------

app = FastAPI(title="Face Search API")

# Will hold embeddings in memory
EMBEDDINGS: List[dict] = []


def load_embeddings_from_s3():
    """Load embeddings.json from S3 into memory."""
    global EMBEDDINGS

    s3 = boto3.client("s3", region_name=AWS_REGION)
    resp = s3.get_object(Bucket=S3_BUCKET, Key=EMBEDDINGS_KEY)
    content = resp["Body"].read()
    data = json.loads(content)

    # Convert embedding lists to numpy arrays for fast math
    for item in data:
        item["embedding"] = np.array(item["embedding"], dtype=np.float32)

    EMBEDDINGS = data
    print(f"Loaded {len(EMBEDDINGS)} embeddings from S3")


def cosine_similarity(a: np.ndarray, b: np.ndarray) -> float:
    return float(np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b)))


@app.on_event("startup")
def startup_event():
    load_embeddings_from_s3()


@app.post("/search-face")
async def search_face(image: UploadFile = File(...)):
    if image.content_type not in ["image/jpeg", "image/png"]:
        raise HTTPException(status_code=400, detail="Image must be JPEG or PNG")

    # Read image bytes into OpenCV format
    contents = await image.read()
    np_arr = np.frombuffer(contents, np.uint8)
    img = cv2.imdecode(np_arr, cv2.IMREAD_COLOR)

    if img is None:
        raise HTTPException(status_code=400, detail="Could not decode image")

    try:
        reps = DeepFace.represent(
            img,
            model_name=MODEL_NAME,
            enforce_detection=True
        )
    except ValueError as e:
        # e.g. no face detected
        raise HTTPException(status_code=400, detail=f"No face detected: {str(e)}")

    query_emb = np.array(reps[0]["embedding"], dtype=np.float32)

    if not EMBEDDINGS:
        raise HTTPException(status_code=500, detail="No embeddings loaded on server")

    # Find best match by cosine similarity
    best_sim: float = -1.0
    best_item: Optional[dict] = None

    for item in EMBEDDINGS:
        sim = cosine_similarity(query_emb, item["embedding"])
        if sim > best_sim:
            best_sim = sim
            best_item = item

    match = best_sim >= SIM_THRESHOLD

    # For security, we just return metadata – not a public S3 URL
    # image_filename is whatever you put in embeddings.json (e.g. "actors/xyz.jpg")
    resp = {
        "match": match,
        "score": best_sim,
        "person_id": best_item["person_id"] if best_item else None,
        "image_filename": best_item["image_filename"] if best_item else None,
    }

    return JSONResponse(content=resp)
