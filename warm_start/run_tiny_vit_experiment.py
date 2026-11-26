#!/usr/bin/env python3
"""Tiny patch-based Vision Transformer CIFAR-10 warm-vs-cold experiment."""

import torch
import torch.nn as nn

from architecture_experiment_common import run_experiment


class TinyVisionTransformer(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.patch_embed = nn.Conv2d(3, 256, kernel_size=4, stride=4)
        self.class_token = nn.Parameter(torch.zeros(1, 1, 256))
        self.position_embedding = nn.Parameter(torch.zeros(1, 65, 256))
        layer = nn.TransformerEncoderLayer(
            d_model=256,
            nhead=8,
            dim_feedforward=1024,
            dropout=0.0,
            activation="gelu",
            batch_first=True,
            norm_first=True,
        )
        self.encoder = nn.TransformerEncoder(layer, num_layers=4)
        self.norm = nn.LayerNorm(256)
        self.head = nn.Linear(256, 10)
        nn.init.trunc_normal_(self.class_token, std=0.02)
        nn.init.trunc_normal_(self.position_embedding, std=0.02)

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        patches = self.patch_embed(inputs).flatten(2).transpose(1, 2)
        cls = self.class_token.expand(inputs.shape[0], -1, -1)
        tokens = torch.cat((cls, patches), dim=1) + self.position_embedding
        encoded = self.encoder(tokens)
        return self.head(self.norm(encoded[:, 0]))


def main() -> None:
    run_experiment(
        "tiny_vit_patch4",
        TinyVisionTransformer,
        {
            "image_size": 32,
            "patch_size": 4,
            "embedding_width": 256,
            "depth": 4,
            "attention_heads": 8,
            "mlp_width": 1024,
            "pretrained": False,
        },
        __file__,
    )


if __name__ == "__main__":
    main()
