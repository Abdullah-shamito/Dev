from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
import torch
from model import SiameseNetwork, prepare_features, predict_synergy  # import your model code
import pandas as pd

app = FastAPI()

# Load your model and features
features_df = pd.read_csv("org_features.csv")
full_features, id_to_idx = prepare_features(features_df)
input_dim = full_features.shape[1]

model = SiameseNetwork(input_dim)
model.load_state_dict(torch.load("siamese_model.pth", map_location="cuda"))
model.eval()

# Request schema
class OrgPair(BaseModel):
    org1_id: str
    org2_id: str

@app.post("/predict_synergy")
def get_synergy_score(pair: OrgPair):
    try:
        score = predict_synergy(pair.org1_id, pair.org2_id, model, full_features, id_to_idx)
        return {"similarity_score": score}
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))
