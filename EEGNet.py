import torch
import torch.nn as nn
import torch.nn.functional as F


class SpatialAttention(nn.Module):
    def __init__(self, channels, embed_dim):
        super().__init__()

        self.query = nn.Linear(channels, embed_dim)
        self.key = nn.Linear(channels, embed_dim)
        self.value = nn.Linear(channels, embed_dim)

        self.scale = embed_dim ** 0.5

    def forward(self, x):

        x = x.permute(0, 3, 2, 1)   # (B, T, C, F)
        x = x.mean(dim=-1)          # (B, T, C)

        Q = self.query(x)           # (B, T, D)
        K = self.key(x)             # (B, T, D)
        V = self.value(x)           # (B, T, D)

        attn = torch.matmul(Q, K.transpose(-2, -1)) / self.scale
        attn = torch.softmax(attn, dim=-1)

        out = torch.matmul(attn, V)   # (B, T, D)

        return out


class EEG_MODEL(nn.Module):
    def __init__(self,
                 num_classes=5,
                 channels=64,
                 F1=8,
                 kernelLength=64,
                 embed_dim=64,
                 dropout=0.5):

        super().__init__()

        self.temporal_conv = nn.Conv2d(
            1, F1,
            (1, kernelLength),
            padding=(0, kernelLength // 2),
            bias=False
        )

        self.bn1 = nn.BatchNorm2d(F1)
        self.elu = nn.ELU()

        self.spatial_attention = SpatialAttention(
            channels=channels,
            embed_dim=embed_dim
        )

        self.pool = nn.AdaptiveAvgPool1d(1)

        self.dropout = nn.Dropout(dropout)

        self.fc = nn.Linear(embed_dim, num_classes)

    def forward(self, x):

        x = self.temporal_conv(x)
        x = self.bn1(x)
        x = self.elu(x)

        x = self.spatial_attention(x)

        x = x.permute(0, 2, 1)   # (B, D, T)
        x = self.pool(x)         # (B, D, 1)
        x = x.squeeze(-1)        # (B, D)

        x = self.dropout(x)

        x = self.fc(x)

        return x
