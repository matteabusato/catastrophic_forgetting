import math

import torch
from torch import Tensor, nn

from src.attention import attention_logits


class Student(nn.Module):
    def __init__(
        self,
        d: int,
        kappa: float,
        use_low_rank: bool = False,
        dtype: torch.dtype = torch.float64,
        device: str = "cpu",
    ):
        super().__init__()

        self.d = d
        self.kappa = kappa
        self.P = int(round(kappa * d))

        self.W = nn.Parameter(torch.randn(d, self.P, dtype=dtype, device=device,) )
        self.w = nn.Parameter(torch.zeros(d,dtype=dtype, device=device,),
            requires_grad=use_low_rank,)
        
        self.use_low_rank = use_low_rank

    @property
    def S_large(self) -> Tensor:
        return self.W @ self.W.T / math.sqrt(self.P)

    @property
    def S(self) -> Tensor:
        S = self.S_large

        if self.use_low_rank:
            S = S + torch.outer(self.w, self.w)

        return S
    
    def attention_logits(self, X: Tensor) -> Tensor:
        return attention_logits(X, self.S)

    def attention(self, X: Tensor) -> Tensor:
        return torch.softmax(self.attention_logits(X), dim=-1)

    def forward(self, X: Tensor) -> Tensor:
        return self.attention(X) @ X

    def set_pretraining_mode(self) -> None:
        self.use_low_rank = False

        self.W.requires_grad_(True)
        self.w.requires_grad_(False)

        with torch.no_grad():
            self.w.zero_()

    def set_finetuning_mode(self) -> None:
        self.use_low_rank = True

        self.W.requires_grad_(False)
        self.w.requires_grad_(True)


    def reset_low_rank(self, scale: float = 1.0, seed: int = 0,) -> None:
        generator = torch.Generator(device=self.w.device)
        generator.manual_seed(seed)

        w_init = scale * torch.randn(self.d, generator=generator, device=self.w.device,
            dtype=self.w.dtype,)

        with torch.no_grad():
            self.w.copy_(w_init)

    def load_large_rank(self, W: Tensor,) -> None:
        with torch.no_grad():
            self.W.copy_(W.to(device=self.W.device, dtype=self.W.dtype,))