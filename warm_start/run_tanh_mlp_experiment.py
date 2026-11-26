#!/usr/bin/env python3
"""Three-layer tanh MLP CIFAR-10 warm-vs-cold experiment."""

import torch.nn as nn

from architecture_experiment_common import run_experiment


class ThreeLayerTanhMLP(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.network = nn.Sequential(
            nn.Flatten(),
            nn.Linear(3 * 32 * 32, 2048),
            nn.Tanh(),
            nn.Linear(2048, 2048),
            nn.Tanh(),
            nn.Linear(2048, 10),
        )

    def forward(self, inputs):
        return self.network(inputs)


def main() -> None:
    run_experiment(
        "three_layer_tanh_mlp",
        ThreeLayerTanhMLP,
        {"hidden_widths": [2048, 2048], "activation": "tanh"},
        __file__,
    )


if __name__ == "__main__":
    main()
