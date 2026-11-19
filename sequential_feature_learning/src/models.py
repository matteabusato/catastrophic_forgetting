from __future__ import annotations

import torch
from torch import nn
from torchvision.models import resnet18


class CIFARResNet18(nn.Module):
    def __init__(self):
        super().__init__()
        net = resnet18(weights=None)
        net.conv1 = nn.Conv2d(3, 64, kernel_size=3, stride=1, padding=1, bias=False)
        net.maxpool = nn.Identity()
        net.fc = nn.Identity()
        self.backbone = net
        self.heads = nn.ModuleDict()

    def add_head(self, task: str, num_classes: int = 5, initial_state=None):
        # Heads are often added after the model has already been moved to CUDA.
        head = nn.Linear(512, num_classes).to(next(self.backbone.parameters()).device)
        if initial_state is not None:
            head.load_state_dict(initial_state)
        self.heads[task] = head
        return head

    def features(self, images):
        return self.backbone(images)

    def forward(self, images, task: str):
        return self.heads[task](self.features(images))


def backbone_state(model: CIFARResNet18):
    return {key: value.detach().cpu().clone() for key, value in model.backbone.state_dict().items()}


def head_states(model: CIFARResNet18):
    return {task: {key: value.detach().cpu().clone() for key, value in head.state_dict().items()}
            for task, head in model.heads.items()}
