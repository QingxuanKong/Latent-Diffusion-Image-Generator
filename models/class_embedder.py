import torch
import torch.nn as nn
import math

class ClassEmbedder(nn.Module):
    def __init__(self, embed_dim, n_classes=1000, cond_drop_rate=0.1):
        super().__init__()
        
        # TODO: implement the class embeddering layer for CFG using nn.Embedding
        self.embedding = nn.Embedding(n_classes, embed_dim) 
        self.cond_drop_rate = cond_drop_rate
        self.num_classes = n_classes
        
        # define a reserved index for unconditional (e.g., last index)
        self.uncond_id = n_classes

        # extend embedding table by 1 for unconditional class
        self.embedding = nn.Embedding(n_classes + 1, embed_dim)

    def forward(self, x):
        """
        x: LongTensor of shape (B,), each element is class index (0 to n_classes - 1)
        """

        b = x.shape[0]
        
        if self.cond_drop_rate > 0 and self.training:
            # create a mask to drop some conditions
            # TODO: implement class drop with unconditional class
            drop_mask = torch.rand(b, device=x.device) < self.cond_drop_rate
            x = x.clone()  # avoid modifying input outside
            x[drop_mask] = self.uncond_id  # replace with "unconditional" index
        
        # TODO: get embedding: N, embed_dim
        # get embeddings: shape [B, embed_dim]
        c = self.embedding(x)
        return c