# features.py
import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler, LabelEncoder
import torch

def load_and_prepare_data(features_df, pairs_df):
    print("Preprocessing features...")

    org_ids = features_df['id'].values if 'id' in features_df.columns else features_df.index
    feature_columns = [col for col in features_df.columns if col != 'id']

    processed_features = features_df[feature_columns].copy()

    categorical_columns = processed_features.select_dtypes(include=['object']).columns
    numerical_columns = processed_features.select_dtypes(include=[np.number]).columns

    label_encoders = {}
    for col in categorical_columns:
        le = LabelEncoder()
        processed_features[col] = le.fit_transform(processed_features[col].astype(str))
        label_encoders[col] = le

    scaler = StandardScaler()
    if len(numerical_columns) > 0:
        processed_features[numerical_columns] = scaler.fit_transform(processed_features[numerical_columns])

    feature_matrix = processed_features.values
    id_to_idx = {org_id: idx for idx, org_id in enumerate(org_ids)}

    print(f"Feature matrix shape: {feature_matrix.shape}")

    valid_pairs = []
    valid_labels = []

    for _, row in pairs_df.iterrows():
        org1_id, org2_id, label = row['org1_id'], row['org2_id'], row['label']
        if org1_id in id_to_idx and org2_id in id_to_idx:
            valid_pairs.append((id_to_idx[org1_id], id_to_idx[org2_id]))
            valid_labels.append(float(label))

    print(f"Valid pairs: {len(valid_pairs)}")

    pairs_array = np.array(valid_pairs)
    labels_array = np.array(valid_labels)

    X1 = feature_matrix[pairs_array[:, 0]]
    X2 = feature_matrix[pairs_array[:, 1]]
    y = labels_array

    return (
        torch.FloatTensor(X1),
        torch.FloatTensor(X2),
        torch.FloatTensor(y),
        feature_matrix,
        id_to_idx,
        scaler,
        label_encoders
    )
