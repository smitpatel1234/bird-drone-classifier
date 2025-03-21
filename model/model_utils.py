import torch
import torch.nn as nn
import numpy as np

class BirdDroneClassifier(nn.Module):
    def __init__(self, input_dim):
        super(BirdDroneClassifier, self).__init__()
        self.model = nn.Sequential(
            nn.Linear(input_dim, 64),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(64, 32),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(32, 1),
            nn.Sigmoid()
        )

    def forward(self, x):
        return self.model(x)

def preprocess_data(data, scaler):
    """
    Preprocess the radar signature data
    
    Args:
        data: The input data from the .pkl file
        scaler: The fitted scaler used during training
    
    Returns:
        Processed data ready for model inference
    """
    # Convert to numpy array if not already
    if not isinstance(data, np.ndarray):
        data = np.array(data)
    
    # Reshape if needed - adjust based on your data format
    if len(data.shape) == 1:
        data = data.reshape(1, -1)
    
    # Apply the same scaling used during training
    return scaler.transform(data)
