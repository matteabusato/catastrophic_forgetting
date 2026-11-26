#!/usr/bin/env python3
"""Small convolutional CIFAR-10 model without residual/skip connections."""

import torch
import torch.nn as nn
import torch.nn.functional as F

from architecture_experiment_common import run_experiment


class ConvBlock(nn.Module):
    def __init__(self, in_channels: int, channels: int, stride: int = 1) -> None:
        super().__init__()
        self.layers = nn.Sequential(
            nn.Conv2d(in_channels, channels, 3, stride=stride, padding=1, bias=False),
            nn.BatchNorm2d(channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(channels, channels, 3, stride=1, padding=1, bias=False),
            nn.BatchNorm2d(channels),
            nn.ReLU(inplace=True),
        )

    def forward(self, inputs):
        return self.layers(inputs)


class PlainCifarCNN(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.stem = nn.Sequential(
            nn.Conv2d(3, 32, 3, stride=1, padding=1, bias=False),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True),
        )
        self.stage1 = ConvBlock(32, 32, stride=1)
        self.stage2 = ConvBlock(32, 64, stride=2)
        self.stage3 = ConvBlock(64, 128, stride=2)
        self.stage4 = ConvBlock(128, 256, stride=2)
        self.classifier = nn.Linear(256, 10)

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        outputs = self.stem(inputs)
        outputs = self.stage1(outputs)
        outputs = self.stage2(outputs)
        outputs = self.stage3(outputs)
        outputs = self.stage4(outputs)
        outputs = F.adaptive_avg_pool2d(outputs, output_size=1).flatten(1)
        return self.classifier(outputs)


def main() -> None:
    run_experiment(
        "plain_cnn_no_skips",
        PlainCifarCNN,
        {"channels": [32, 64, 128, 256], "conv_blocks": 4, "skip_connections": False},
        __file__,
    )


if __name__ == "__main__":
    main()
