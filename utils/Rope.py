import torch
import torch.nn as nn

class PositionalExprEmbedding(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.mask_token_id = -10
        self.inv_freq = nn.Parameter(
            1. / (100 ** (torch.arange(0, dim, 2).float() / dim)),
            requires_grad=False
        )

    def forward(self, x):
        x_mask_idx = (x == self.mask_token_id).nonzero(as_tuple=False)
        x = torch.einsum("bi,j->bij", x, self.inv_freq)
        x = torch.cat((x.sin(), x.cos()), dim=-1)
        if x_mask_idx.numel() > 0:
            x[x_mask_idx[:, 0], x_mask_idx[:, 1]] = 0
        return x