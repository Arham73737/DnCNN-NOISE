"""
DnCNN architecture (residual learning image denoiser).
Reference: Zhang et al., "Beyond a Gaussian Denoiser: Residual Learning of
Deep CNN for Image Denoising", 2017.

The network predicts the residual noise present in the image:
clean = noisy - predicted_noise.
"""

import torch.nn as nn


class DnCNN(nn.Module):
    def __init__(self, in_nc=3, out_nc=3, nc=64, nb=20):
        super(DnCNN, self).__init__()

        layers = [
            nn.Conv2d(in_nc, nc, kernel_size=3, stride=1, padding=1, bias=True),
            nn.ReLU(inplace=True),
        ]

        for _ in range(nb - 2):
            layers.append(nn.Conv2d(nc, nc, kernel_size=3, stride=1, padding=1, bias=True))
            layers.append(nn.ReLU(inplace=True))

        layers.append(nn.Conv2d(nc, out_nc, kernel_size=3, stride=1, padding=1, bias=True))

        self.model = nn.Sequential(*layers)

    def forward(self, x):
        predicted_noise = self.model(x)
        return x - predicted_noise
