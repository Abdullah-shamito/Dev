import runpod
from api import load_model_and_inference, predict_synergy

# Load model and functions (one-time on cold start)
load_model_and_inference()

def handler(event):
    try:
        input_data = event["input"]
        org1_id = input_data["org1_id"]
        org2_id = input_data["org2_id"]
        score = predict_synergy(org1_id, org2_id)
        return {
            "org1_id": org1_id,
            "org2_id": org2_id,
            "synergy_score": score
        }
    except Exception as e:
        return {"error": str(e)}

# Required by RunPod Serverless
runpod.serverless.start({"handler": handler})
