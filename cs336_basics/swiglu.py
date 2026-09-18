import torch
import torch.nn as nn

class SwiGLU(torch.nn.Module):
    def __init__(self, d_model, d_ff):
        super().__init__()
        self.w1 = nn.Parameter(torch.empty(d_ff,d_model))
        self.w2 = nn.Parameter(torch.empty((d_model,d_ff)))
        self.w3 = nn.Parameter(torch.empty((d_ff,d_model)))
        
        
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # w2 dot (SiLU (w1 dot x) dim dot (w3 dot x))
        y1 = x @ self.w1.T
        y1 = y1 * torch.sigmoid(y1)
        y3 = x @ self.w3.T
        return (y1 * y3) @ self.w2.T