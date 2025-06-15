import torch
import torch.nn as nn
import pandas as pd
import numpy as np
import ast
from sklearn.preprocessing import MultiLabelBinarizer, MinMaxScaler

# --- Siamese Network Definition (MUST match training exactly!) ---
class SiameseNetwork(nn.Module):
    def __init__(self, input_dim):  # Fixed __init__
        super(SiameseNetwork, self).__init__()
        self.fc = nn.Sequential(
            nn.Linear(input_dim, 256),  # Match training architecture
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(256, 128),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Linear(64, 32)
        )
        
    def forward_once(self, x):  # Match training method name
        return self.fc(x)
    
    def forward(self, x1, x2):
        out1 = self.forward_once(x1)
        out2 = self.forward_once(x2)
        return torch.cosine_similarity(out1, out2, dim=1)  # Match training

# --- Feature Processing (replicate training preprocessing) ---
def safe_parse(val):
    try:
        return ast.literal_eval(val) if isinstance(val, str) else []
    except:
        return []

def prepare_features_for_inference(features_df):
    """
    Prepare features exactly as done in training
    """
    # Make a copy to avoid modifying original
    df = features_df.copy().set_index('id')
    
    # Clean and normalize binary flags
    binary_cols = ['is_network', 'is_global', 'is_enriched']
    for col in binary_cols:
        df[col] = df[col].map({'t': 1, 'f': 0, True: 1, False: 0}).fillna(0).astype(int)

    # Parse list columns
    for col in ['categories', 'sdgs', 'active_regions']:
        df[col] = df[col].fillna('[]').apply(safe_parse)

    # One-hot encode list fields
    mlb_cat = MultiLabelBinarizer()
    cat_ohe = pd.DataFrame(mlb_cat.fit_transform(df['categories']), index=df.index)

    mlb_sdg = MultiLabelBinarizer()
    sdg_ohe = pd.DataFrame(mlb_sdg.fit_transform(df['sdgs']), index=df.index)

    mlb_reg = MultiLabelBinarizer()
    region_ohe = pd.DataFrame(mlb_reg.fit_transform(df['active_regions']), index=df.index)

    # Normalize continuous numeric features
    numeric_cols = [
        'fundraising_score', 'investor_score', 'maturity_score', 'impact_score',
        'climate_score', 'linkedin_follower_count', 'type_confidence'
    ]
    scaler = MinMaxScaler()
    scaled_numeric = pd.DataFrame(
        scaler.fit_transform(df[numeric_cols].fillna(0)),
        index=df.index,
        columns=numeric_cols
    )

    # Combine all features (excluding embeddings for now)
    combined_df = pd.concat([
        scaled_numeric,
        df[binary_cols],
        cat_ohe,
        sdg_ohe,
        region_ohe
    ], axis=1)

    # Convert embeddings (stored as stringified arrays) to vectors
    embedding_vectors = df['embedding'].apply(lambda x: np.array(ast.literal_eval(x)) if isinstance(x, str) else np.zeros(384))
    embedding_matrix = np.stack(embedding_vectors.values)

    # Combine everything
    full_features = np.hstack([embedding_matrix, combined_df.values])
    id_to_idx = {org_id: idx for idx, org_id in enumerate(df.index)}
    
    return torch.tensor(full_features, dtype=torch.float32), id_to_idx

# --- Load and prepare features ---
print("Loading features...")
features_df = pd.read_csv("org_features.csv")
full_features, id_to_idx = prepare_features_for_inference(features_df)

print(f"Loaded {len(id_to_idx)} organizations with {full_features.shape[1]} features each")

# --- Load model ---
print("Loading trained model...")
input_dim = full_features.shape[1]
model = SiameseNetwork(input_dim)
model.load_state_dict(torch.load("siamese_model.pth", map_location='cpu'))
model.eval()

print(f"Model loaded with input dimension: {input_dim}")

# --- Prediction Function ---
def predict_synergy(org1_id, org2_id):
    """
    Predict synergy/similarity score between two organizations
    Returns a score between -1 and 1 (cosine similarity)
    """
    if org1_id not in id_to_idx:
        raise ValueError(f"Organization ID '{org1_id}' not found in feature data.")
    if org2_id not in id_to_idx:
        raise ValueError(f"Organization ID '{org2_id}' not found in feature data.")
    
    # Get feature vectors
    idx1 = id_to_idx[org1_id]
    idx2 = id_to_idx[org2_id]
    
    x1 = full_features[idx1].unsqueeze(0)  # Add batch dimension
    x2 = full_features[idx2].unsqueeze(0)  # Add batch dimension
    
    with torch.no_grad():
        similarity = model(x1, x2)
        score = similarity.item()  # Convert to Python float
    
    return score

def predict_synergy_batch(org_pairs):
    """
    Predict synergy for multiple organization pairs
    org_pairs: list of tuples [(org1_id, org2_id), ...]
    Returns: list of scores
    """
    results = []
    for org1_id, org2_id in org_pairs:
        try:
            score = predict_synergy(org1_id, org2_id)
            results.append((org1_id, org2_id, score))
        except ValueError as e:
            results.append((org1_id, org2_id, None))
            print(f"Error predicting {org1_id} vs {org2_id}: {e}")
    
    return results

def find_most_similar(target_org_id, top_k=10):
    """
    Find organizations most similar to the target organization
    """
    if target_org_id not in id_to_idx:
        raise ValueError(f"Organization ID '{target_org_id}' not found in feature data.")
    
    similarities = []
    target_idx = id_to_idx[target_org_id]
    target_features = full_features[target_idx].unsqueeze(0)
    
    with torch.no_grad():
        for org_id, idx in id_to_idx.items():
            if org_id != target_org_id:
                org_features = full_features[idx].unsqueeze(0)
                similarity = model(target_features, org_features).item()
                similarities.append((org_id, similarity))
    
    # Sort by similarity (highest first)
    similarities.sort(key=lambda x: x[1], reverse=True)
    
    return similarities[:top_k]

print("Inference module loaded successfully!")
print("Available functions:")
print("- predict_synergy(org1_id, org2_id)")
print("- predict_synergy_batch(org_pairs)")
print("- find_most_similar(target_org_id, top_k=10)")