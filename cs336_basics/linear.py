import torch
import torch.nn as nn
from torch.nn.init import trunc_normal_

class Linear(nn.Module):
    def __init__(self, in_features, out_features, device=None, dtype=None):
        super().__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.weight = nn.Parameter(torch.empty((out_features, in_features), device=device, dtype=dtype))
        sigma = torch.sqrt(2 / (in_features + out_features))
        trunc_normal_(self.weight, mean=0.0, std = sigma, a=-3*sigma, b=3*sigma)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return torch.einsum("...j,ij->...i",x,self.weight)