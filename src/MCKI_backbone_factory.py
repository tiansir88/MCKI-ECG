from typing import Dict, Optional, Tuple

import torch
import torch.nn as nn

from train_v3_2stage import TwoStageModel


SUPPORTED_MCKI_BACKBONES = {'resnet18', 'default', 'two_stage_resnet18'}


def _infer_backbone_dim(model: nn.Module, device: torch.device) -> int:
    with torch.no_grad():
        dummy_in = torch.randn(1, 12, 1000, device=device)
        dummy_out = model.backbone(dummy_in)
        feat = dummy_out[0] if isinstance(dummy_out, tuple) else dummy_out
    return int(feat.shape[1])


def build_MCKI_backbone(
    backbone_name: str,
    num_classes: int,
    device: torch.device,
    cfg: Optional[Dict] = None,
) -> Tuple[nn.Module, int]:
    cfg = cfg or {}
    normalized_name = str(backbone_name).lower().strip()

    if normalized_name not in SUPPORTED_MCKI_BACKBONES:
        raise ValueError(
            f'Phase-1 MCKI backbone factory only supports {sorted(SUPPORTED_MCKI_BACKBONES)}. '
            f'Got: {backbone_name}'
        )

    projection_dim = int(cfg.get('proj_dim', 128))
    model = TwoStageModel(num_classes=num_classes, projection_dim=projection_dim).to(device)
    backbone_dim = _infer_backbone_dim(model, device)
    return model, backbone_dim
