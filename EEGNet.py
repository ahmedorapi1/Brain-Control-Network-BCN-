import torch
import torch.nn as nn
import torch.nn.functional as F


class SpatialAttention(nn.Module):
    def __init__(self, in_dim, embed_dim):
        super().__init__()

        self.query = nn.Linear(in_dim, embed_dim)
        self.key = nn.Linear(in_dim, embed_dim)
        self.value = nn.Linear(in_dim, embed_dim)

        self.scale = embed_dim ** 0.5
        self.norm = nn.LayerNorm(embed_dim)

    def forward(self, x):
        # x: (B, 8, 1, 50)

        x = x.squeeze(2)          # (B, 8, 50)
        x = x.permute(0, 2, 1)    # (B, 50, 8)

        Q = self.query(x)
        K = self.key(x)
        V = self.value(x)

        attn = torch.matmul(Q, K.transpose(-2, -1)) / self.scale
        attn = torch.softmax(attn, dim=-1)

        out = torch.matmul(attn, V)   # (B, 50, 64)

        return self.norm(out)


class EEG_MODEL(nn.Module):
    def __init__(self,
                 num_classes=4,
                 channels=22,
                 F1=8,
                 kernelLength=64,
                 embed_dim=64,
                 dropout=0.25):
        super().__init__()

        self.temporal_conv = nn.Conv2d(
            1, F1,
            (1, kernelLength),
            padding=(0, kernelLength // 2),
            bias=False  # (B, 8, 22, 1000)
        )
        self.pool1 = nn.AvgPool2d((1, 20))  # (B, 8, 22, 50)

        self.bn1 = nn.BatchNorm2d(F1)

        self.spatial_conv = nn.Conv2d(
            F1, F1,
            (channels, 1),  # (B, 8, 1, 50)
            groups=F1,
            bias=False
        )

        self.bn2 = nn.BatchNorm2d(F1)
        self.elu = nn.ELU()

        self.attn = SpatialAttention(
            in_dim=F1,
            embed_dim=embed_dim
        )

        self.pool2 = nn.AdaptiveAvgPool1d(4)
        self.dropout = nn.Dropout(dropout)

        self.fc = nn.Linear(embed_dim * 4, num_classes)

    def forward(self, x):
        # x: (B, 1, C, T)

        x = self.temporal_conv(x)

        x = self.bn1(x)
        x = self.elu(x)
        x = self.pool1(x)

        x = self.spatial_conv(x)
        x = self.bn2(x)
        x = self.elu(x)

        x = self.attn(x)  # (B, 50,64)

        x = x.permute(0, 2, 1)  # (B, 64, 50)

        x = self.pool2(x)  # (B, 64, 4)

        x = x.flatten(1)  # (B, 256)

        x = self.dropout(x)
        x = self.fc(x)

        return x