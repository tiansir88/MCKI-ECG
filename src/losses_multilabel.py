import torch
import torch.nn as nn


class AsymmetricLossMultiLabel(nn.Module):
    """A simple, stable ASL implementation for multilabel classification."""

    def __init__(self, gamma_neg: float = 4.0, gamma_pos: float = 1.0, clip: float = 0.05, eps: float = 1e-8):
        super().__init__()
        self.gamma_neg = float(gamma_neg)
        self.gamma_pos = float(gamma_pos)
        self.clip = float(clip)
        self.eps = float(eps)

    def forward(self, logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        targets = targets.float()
        xs_pos = torch.sigmoid(logits)
        xs_neg = 1.0 - xs_pos

        if self.clip is not None and self.clip > 0:
            xs_neg = torch.clamp(xs_neg + self.clip, max=1.0)

        log_pos = torch.log(torch.clamp(xs_pos, min=self.eps, max=1.0))
        log_neg = torch.log(torch.clamp(xs_neg, min=self.eps, max=1.0))

        loss = targets * log_pos + (1.0 - targets) * log_neg

        if self.gamma_neg > 0 or self.gamma_pos > 0:
            pt = xs_pos * targets + xs_neg * (1.0 - targets)
            gamma = self.gamma_pos * targets + self.gamma_neg * (1.0 - targets)
            one_sided_w = torch.pow(1.0 - pt, gamma)
            loss = loss * one_sided_w

        return -loss.mean()
