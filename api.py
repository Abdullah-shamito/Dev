from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
import sys
import os

# Global variables to hold the imported functions
predict_synergy = None
predict_synergy_batch = None 
find_most_similar = None

app = FastAPI(title="Siamese Network API - Explicit Version")

class SynergyRequest(BaseModel):
    org1_id: str  # UUID support
    org2_id: str

class BatchRequest(BaseModel):
    org_pairs: list

class SimilarityRequest(BaseModel):
    target_org_id: str
    top_k: int = 10

@app.get("/")
def root():
    return {"message": "Siamese Network API", "status": "running"}

@app.get("/health")
def health_check():
    return {
        "status": "healthy",
        "message": "API is running",
        "model_loaded": predict_synergy is not None
    }

@app.get("/predict-get")
def predict_synergy_get(org1_id: str, org2_id: str):
    if predict_synergy is None:
        raise HTTPException(status_code=503, detail="Model not loaded")
    try:
        score = predict_synergy(org1_id, org2_id)
        return {
            "org1_id": org1_id,
            "org2_id": org2_id,
            "synergy_score": score,
            "method": "GET"
        }
    except Exception as e:
        raise HTTPException(status_code=400, detail=str(e))

@app.post("/predict")
def predict_synergy_endpoint(request: SynergyRequest):
    if predict_synergy is None:
        raise HTTPException(status_code=503, detail="Model not loaded")
    try:
        score = predict_synergy(request.org1_id, request.org2_id)
        return {
            "org1_id": request.org1_id,
            "org2_id": request.org2_id,
            "synergy_score": score
        }
    except Exception as e:
        raise HTTPException(status_code=400, detail=str(e))

@app.post("/predict/batch")
def predict_batch_endpoint(request: BatchRequest):
    if predict_synergy_batch is None:
        raise HTTPException(status_code=503, detail="Model not loaded")
    try:
        org_pairs = [(pair[0], pair[1]) for pair in request.org_pairs]
        results = predict_synergy_batch(org_pairs)
        return {"results": results}
    except Exception as e:
        raise HTTPException(status_code=400, detail=str(e))

@app.post("/similarity")
def find_similar_endpoint(request: SimilarityRequest):
    if find_most_similar is None:
        raise HTTPException(status_code=503, detail="Model not loaded")
    try:
        similar_orgs = find_most_similar(request.target_org_id, request.top_k)
        return {
            "target_org_id": request.target_org_id,
            "similar_organizations": [
                {"org_id": org_id, "similarity_score": score}
                for org_id, score in similar_orgs
            ]
        }
    except Exception as e:
        raise HTTPException(status_code=400, detail=str(e))

def load_model_and_inference():
    """Explicitly load the model and inference functions"""
    global predict_synergy, predict_synergy_batch, find_most_similar

    print("🔍 Checking required files...")
    required_files = ['org_features.csv', 'siamese_model.pth', 'inference.py', 'features.py']
    missing_files = [f for f in required_files if not os.path.exists(f)]

    if missing_files:
        print(f"❌ Missing files: {missing_files}")
        return False

    print("✅ All required files found")

    if '.' not in sys.path:
        sys.path.insert(0, '.')

    try:
        from inference import predict_synergy as ps, predict_synergy_batch as psb, find_most_similar as fms
        predict_synergy = ps
        predict_synergy_batch = psb
        find_most_similar = fms

        print("✅ Model and inference functions loaded successfully!")

        import pandas as pd
        features_df = pd.read_csv("org_features.csv")
        sample_org_ids = features_df['id'].head(3).tolist()
        if len(sample_org_ids) >= 2:
            test_score = predict_synergy(sample_org_ids[0], sample_org_ids[1])
            print(f"🔎 Test prediction: {test_score:.4f}")

        return True
    except Exception as e:
        print(f"❌ Error loading model: {e}")
        return False

# 🚀 Load model immediately on import (for RunPod Serverless)
print("🚀 Initializing Siamese Network API (serverless mode)")
print("=" * 50)
load_model_and_inference()
