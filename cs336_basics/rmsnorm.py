import torch
import torch.nn as nn

class RMSNorm(nn.Module):
    def __init__(self, d_model: int, eps: float = 1e-5, device=None, dtype=None):
        super().__init__()
        self.d_model = d_model
        self.eps = eps
        self.device = device
        self.dtype = dtype
        self.weight = nn.Parameter(torch.empty((d_model),device=device,dtype=dtype))
        
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # (batch_size, sequence_length, d_model) 
        x = x.to(torch.float32)
        
        rms = torch.sqrt(
            x.pow(2).mean(dim=-1,keepdim =True) + self.eps
        )
        
        x = x / rms
        
        res = torch.einsum("bsd,d->bsd",x, self.weight)
        
        return res.to(self.dtype)