
import torch
import numpy as np

def predict_radar_signature(sample, model_path='model.pth', scaler_path='scaler.joblib'):
    # Load model and scaler
    model = BirdDroneClassifier(sample.shape[1])
    model.load_state_dict(torch.load(model_path))
    model.eval()

    scaler = joblib.load(scaler_path)

    # Prepare input
    sample_scaled = scaler.transform(sample.reshape(1, -1))
    input_tensor = torch.FloatTensor(sample_scaled)

    # Make prediction
    with torch.no_grad():
        output = model(input_tensor)
        pred_class = 'Drone' if output.item() > 0.5 else 'Bird'
        confidence = output.item() if output.item() > 0.5 else 1 - output.item()

    return pred_class, confidence
