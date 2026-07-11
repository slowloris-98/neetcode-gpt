import torch
import torch.nn as nn
from torchtyping import TensorType

class Solution(nn.Module):
    def __init__(self):
        super().__init__()
        torch.manual_seed(0)
        # Architecture: Linear(784, 512) -> ReLU -> Dropout(0.2) -> Linear(512, 10) -> Sigmoid
        self.linear1 = nn.Linear(784,512)
        self.relu = nn.ReLU()
        self.dropout = nn.Dropout(p=0.2)
        self.projection = nn.Linear(512,10)
        self.sigmoid = nn.Sigmoid()


        

    def forward(self, images: TensorType[float]) -> TensorType[float]:
        torch.manual_seed(0)
        # images shape: (batch_size, 784)
        # Return the model's prediction to 4 decimal places
        x1 = self.linear1(images)
        a1 = self.relu(x1)
        a1_hat = self.dropout(a1) 
        y1 = self.projection(a1_hat)
        y1_hat = self.sigmoid(y1)

        return torch.round(y1_hat,decimals=4)

        
