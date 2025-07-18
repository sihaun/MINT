import torch.nn as nn

# 간단한 Image Encoder (작게)
class SimpleImageEncoder(nn.Module):
    def __init__(self, output_dim=512):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv2d(3, 32, 3, 2, 1),  # 32x16x16
            nn.ReLU(),
            nn.Conv2d(32, 64, 3, 2, 1),  # 64x8x8
            nn.ReLU(),
            nn.AdaptiveAvgPool2d((1, 1)),
            nn.Flatten(),
            nn.Linear(64, output_dim)
        )

    def forward(self, x):
        return self.net(x)