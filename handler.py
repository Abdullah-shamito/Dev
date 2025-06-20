from api import load_model_and_inference, predict_synergy

# Ensure model is loaded at cold start
load_model_and_inference()

def handler(event):
    """
    RunPod Serverless handler for predict_synergy
    """
    try:
        org1_id = event['input']['org1_id']
        org2_id = event['input']['org2_id']
        score = predict_synergy(org1_id, org2_id)
        return {
            "org1_id": org1_id,
            "org2_id": org2_id,
            "synergy_score": score
        }
    except Exception as e:
        return {"error": str(e)}
