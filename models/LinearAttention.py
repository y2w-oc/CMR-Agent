import torch
import torch.nn as nn

def elu_feature_map(x):
    return torch.nn.functional.elu(x) + 1


class LinearAttention(nn.Module):
    def __init__(self, d_model=64, nhead=6, eps=1e-6):
        super().__init__()
        self.feature_map = elu_feature_map
        self.eps = eps

        self.dim = d_model // nhead
        self.nhead = nhead

        # multi-head attention
        self.q_proj = nn.Linear(d_model, d_model, bias=False)
        self.k_proj = nn.Linear(d_model, d_model, bias=False)
        self.v_proj = nn.Linear(d_model, d_model, bias=False)
        self.merge = nn.Linear(d_model, d_model, bias=False)

        # feed-forward network
        self.mlp = nn.Sequential(
            nn.Linear(d_model * 2, d_model * 2, bias=False),
            nn.ReLU(True),
            nn.Dropout(0.1),
            nn.Linear(d_model * 2, d_model, bias=False),
            nn.Dropout(0.1),
        )

        # norm and dropout
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)

        self.att_dropout = nn.Dropout(0.1)

    def forward(self, x, y):
        """
        Params:
            x (torch.Tensor): [N, L, C]
            y (torch.Tensor): [N, S, C]
        """
        bs = x.size(0)
        query, key, value = x, y, y

        # multi-head attention
        query = self.q_proj(query).view(bs, -1, self.nhead, self.dim)  # [N, L, (H, D)]
        key = self.k_proj(key).view(bs, -1, self.nhead, self.dim)  # [N, S, (H, D)]
        value = self.v_proj(value).view(bs, -1, self.nhead, self.dim)
        # message = self.attention(query, key, value, q_mask=x_mask, kv_mask=source_mask)  # [N, L, (H, D)]

        Q = self.feature_map(query)
        K = self.feature_map(key)

        v_length = value.size(1)
        value = value / v_length  # prevent fp16 overflow
        KV = torch.einsum("nshd,nshv->nhdv", K, value)  # (S,D)' @ S,V
        Z = 1 / (torch.einsum("nlhd,nhd->nlh", Q, K.sum(dim=1)) + self.eps)
        message = torch.einsum("nlhd,nhdv,nlh->nlhv", Q, KV, Z) * v_length

        message = message.contiguous()
        message = self.merge(message.view(bs, -1, self.nhead * self.dim))  # [N, L, C]
        message = self.norm1(message)

        message = self.att_dropout(message)

        # feed-forward network
        message = self.mlp(torch.cat([x, message], dim=2))
        message = self.norm2(message)

        attention_output = x + message

        return attention_output