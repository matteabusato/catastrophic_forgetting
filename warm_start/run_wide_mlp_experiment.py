#!/usr/bin/env python3
"""Wide ReLU MLP CIFAR-10 warm-vs-cold experiment."""

import torch.nn as nn

from architecture_experiment_common import run_experiment


class WideReLUMlp(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.network = nn.Sequential(
            nn.Flatten(),
            nn.Linear(3 * 32 * 32, 4096),
            nn.ReLU(inplace=True),
            nn.Linear(4096, 4096),
            nn.ReLU(inplace=True),
            nn.Linear(4096, 10),
        )

    def forward(self, inputs):
        return self.network(inputs)


def main() -> None:
    run_experiment(
        "wide_relu_mlp",
        WideReLUMlp,
        {"hidden_widths": [4096, 4096], "activation": "relu"},
        __file__,
    )


if __name__ == "__main__":
    main()
