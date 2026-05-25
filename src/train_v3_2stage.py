''''''
'''方法说明 + 模型定义'''
import torch
import torch.nn as nn
import torch.nn.functional as F

from resnet1d import resnet18


class TwoStageModel(nn.Module):
    """
    Two-stage MCKI backbone/classifier definition.

    Stage 1 concept:
        The backbone features can be sent to a projection head for contrastive
        pretraining.

    Stage 2 concept:
        The backbone features are passed to a linear classification head for
        downstream multilabel ECG classification.

    Note:
        This file only defines the model structure for documentation and reuse.
        The actual project training/evaluation protocols are implemented in
        run_master_evaluation2.py.
    """

    def __init__(self, num_classes: int = 5, projection_dim: int = 128):
        super().__init__()
        self.backbone = resnet18(num_classes=num_classes)
        num_ftrs = self.backbone.fc.in_features
        self.backbone.fc = nn.Identity()

        # Kept for architectural completeness/documentation.
        self.projection_head = nn.Sequential(
            nn.Linear(num_ftrs, 128),
            nn.ReLU(),
            nn.Linear(128, projection_dim),
        )
        self.cls_head = nn.Linear(num_ftrs, num_classes)

    def forward_backbone(self, x: torch.Tensor) -> torch.Tensor:
        feat = self.backbone(x)
        if isinstance(feat, tuple):
            feat = feat[0]
        return feat

    def forward_contrastive(self, x: torch.Tensor) -> torch.Tensor:
        feat = self.forward_backbone(x)
        return F.normalize(self.projection_head(feat), dim=1)

    def forward_cls(self, x: torch.Tensor) -> torch.Tensor:
        feat = self.forward_backbone(x)
        return self.cls_head(feat)

    def forward(self, x: torch.Tensor, mode: str = 'cls') -> torch.Tensor:
        if mode == 'backbone':
            return self.forward_backbone(x)
        if mode == 'contrastive':
            return self.forward_contrastive(x)
        if mode == 'cls':
            return self.forward_cls(x)
        raise ValueError("mode must be one of {'backbone', 'contrastive', 'cls'}")
