# explicit_app.py - Shows exactly how imports work
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
import uvicorn
import sys
import os

# Global variables to hold the imported functions
predict_synergy = None
predict_synergy_batch = None 
find_most_similar = None

app = FastAPI(title="Siamese Network API - Explicit Version")

class SynergyRequest(BaseModel):
    org1_id: str  # Changed from int to str for UUID support
    org2_id: str  # Changed from int to str for UUID support

class BatchRequest(BaseModel):
    org_pairs: list

class SimilarityRequest(BaseModel):
    target_org_id: str  # Changed from int to str for UUID support
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

# GET endpoint for easy browser testing (added)
@app.get("/predict-get")
def predict_synergy_get(org1_id: str, org2_id: str):
    """Predict synergy via GET request (browser-friendly)"""
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
    """Predict synergy between two organizations"""
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
    """Batch prediction for multiple pairs"""
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
    """Find most similar organizations"""
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
    
    print(" Checking required files...")
    
    # Check if required files exist
    required_files = ['org_features.csv', 'siamese_model.pth', 'inference.py', 'features.py']
    missing_files = []
    
    for file in required_files:
        if not os.path.exists(file):
            missing_files.append(file)
    
    if missing_files:
        print(f"❌ Missing files: {missing_files}")
        print("Please ensure all required files are in the current directory:")
        for file in required_files:
            status = "✅" if file not in missing_files else "❌"
            print(f"  {status} {file}")
        return False
    
    print(" All required files found")
    
    # Add current directory to Python path (in case of import issues)
    if '.' not in sys.path:
        sys.path.insert(0, '.')
    
    try:
        print(" Loading inference functions...")
        print("   This will trigger:")
        print("   1. Loading org_features.csv")
        print("   2. Processing features")
        print("   3. Loading siamese_model.pth")
        print("   4. Setting up model for inference")
        
        # This is where the magic happens - importing triggers everything in inference.py
        from inference import predict_synergy as ps, predict_synergy_batch as psb, find_most_similar as fms
        
        # Assign to global variables
        predict_synergy = ps
        predict_synergy_batch = psb
        find_most_similar = fms
        
        print(" Model and inference functions loaded successfully!")
        
        # Test that functions work
        print(" Testing functions...")
        
        # Get some org IDs to test with
        import pandas as pd
        features_df = pd.read_csv("org_features.csv")
        sample_org_ids = features_df['id'].head(3).tolist()
        
        if len(sample_org_ids) >= 2:
            test_score = predict_synergy(sample_org_ids[0], sample_org_ids[1])
            print(f" Test prediction: {test_score:.4f}")
        
        return True
        
    except Exception as e:
        print(f" Error loading model: {e}")
        print(f" Error type: {type(e).__name__}")
        print(f" Error details: {str(e)}")
        
        # Give specific guidance based on error type
        if "No module named" in str(e):
            print(" This looks like a missing dependency. Try:")
            print("   pip install torch pandas scikit-learn numpy")
        elif "FileNotFoundError" in str(e):
            print(" This looks like a missing file. Check:")
            print("   - org_features.csv exists and is readable")
            print("   - siamese_model.pth exists and is readable")
        elif "dimension mismatch" in str(e).lower():
            print(" This looks like a model architecture mismatch.")
            print("   Your current features don't match the trained model.")
        
        return False

if __name__ == "__main__":
    print(" Starting Siamese Network API...")
    print("=" * 50)
    
    # Explicitly load model
    if load_model_and_inference():
        print("=" * 50)
        print("🎉 Starting API server...")
        uvicorn.run(app, host="0.0.0.0", port=8000)
    else:
        print("=" * 50)
        print(" Failed to load model. Please fix the issues above.")
        print(" Common solutions:")
        print("   1. Make sure all files are in the same directory")
        print("   2. Install dependencies: pip install -r requirements.txt")
        print("   3. Check that your inference.py works standalone")
        sys.exit(1)
