import torch
import torch.nn as nn
from torch.nn.init import trunc_normal_

class Embedding(nn.Module):
    
    def __init__(self, num_embeddings, embedding_dim, device=None, dtype=None):
        super().__init__()
        self.num_embeddings = num_embeddings
        self.embedding_dim = embedding_dim
        self.weight = nn.Parameter(torch.empty((num_embeddings, embedding_dim), device=device, dtype=dtype))
        trunc_normal_(self.weight,0,1,-3,3)
    
    def forward(self, token_ids: torch.Tensor):
        return self.weight[token_ids]