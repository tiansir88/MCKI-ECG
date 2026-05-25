import copy
from pathlib import Path
import json
import os
import random
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
from tqdm import tqdm

import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from sklearn.metrics import accuracy_score, average_precision_score, f1_score, roc_auc_score
from torch.utils.data import DataLoader, Subset  # 新增 Subset 和 DataLoader 导入

from dataset_v3 import get_dataloader_v3
from MCKI_backbone_factory import build_MCKI_backbone
from MCKI_loss_pro import MCKILossPro
from MCKI_relation_builder_stage4 import (
    CLASS_NAMES,
    blend_relation_matrices,
    estimate_confusion_matrix_from_probs,
    load_prior_matrix,
    save_relation_artifacts,
)

REPO_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_DATA_DIR = str(REPO_ROOT / "data" / "processed_v3")
DEFAULT_SAVE_DIR = str(REPO_ROOT / "outputs" / "stage8_leadaware_multiscale_MCKI_5protocols")
DEFAULT_HARD_PAIRS = str(REPO_ROOT / "resources" / "confusable_pairs_v1.csv")
# 修改为 5 协议
DEFAULT_PROTOCOLS = ['Few_Shot_1%', 'Few_Shot_10%', 'Few_Shot_25%']
DEFAULT_SEEDS = [42, 123, 1024]

# 完全保留 Stage 8 的专属超参数（无 beat-aware，lead_mask_prob 为 0.90）
CFG = {
    'backbone_name': 'resnet18',
    'batch_size': 64,
    'num_workers': 4,
    'pin_memory': True,
    'temperature': 0.1,
    'alpha': 2.0,
    'hard_negative_threshold': 0.10,
    'use_continuous_weights': False,
    'noise_std': 0.02,
    'use_polywindow': True,
    'polywindow_lengths': [250, 500, 750],
    'pretrain_epochs': 40,
    'warmup_pretrain_epochs': 10,
    'pretrain_lr': 1e-4,
    'pretrain_weight_decay': 1e-5,
    'proj_dim': 128,
    'bootstrap_lp_epochs': 8,
    'bootstrap_lp_lr': 1e-3,
    'bootstrap_lp_weight_decay': 1e-4,
    'lambda_prior': 0.5,
    'lambda_conf': 0.5,
    'finetune_epochs': 40,
    'lp_epochs': 25,
    'ft_lr': 1e-4,
    'ft_weight_decay': 1e-5,
    'backbone_lr_mult': 0.1,
    'head_lr_mult': 1.0,
    'lp_lr': 1e-3,
    'lp_weight_decay': 1e-4,
    'monitor_metric': 'AUPRC',
    'early_stop_patience': 8,
    'scheduler': 'cosine',
    'tune_thresholds': True,
    'use_pos_weight': True,
    'relation_matrix_values': None,
    # ===== Stage6 Multi-scale local branch =====
    'use_multiscale_local_branch': True,
    'local_window_size': 200,
    'local_num_windows': 4,
    'local_jitter': 12,
    'local_loss_weight': 0.30,
    'align_loss_weight': 0.20,
    # ===== Stage8 Lead-aware + dynamic lead masking =====
    # Best refined config: lmp0p9_md3
    'use_lead_aware_input': True,
    'lead_emb_dim': 16,
    'lead_adapter_dropout': 0.0,
    'use_dynamic_lead_mask': True,
    'lead_mask_prob': 0.90,
    'lead_mask_min_drop': 1,
    'lead_mask_max_drop': 3,
    'log_pretrain_every': 5,
}


def seed_everything(seed: int):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def ensure_dir(path: str):
    os.makedirs(path, exist_ok=True)


# ==========================================
# 新增：支持 Few-Shot 数据集动态提取的工具函数
# ==========================================
def parse_few_shot_ratio(protocol: str) -> Optional[float]:
    if not protocol.startswith('Few_Shot_'):
        return None
    suffix = protocol[len('Few_Shot_'):].strip()
    if not suffix.endswith('%'):
        raise ValueError(f'无法解析 few-shot 协议比例: {protocol}')
    ratio = float(suffix[:-1]) / 100.0
    if not (0.0 < ratio <= 1.0):
        raise ValueError(f'few-shot 比例必须在 (0, 1] 内，当前为: {ratio}')
    return ratio


def build_loader(dataset, batch_size: int, shuffle: bool, num_workers: int, seed: Optional[int] = None):
    generator = None
    if shuffle:
        generator = torch.Generator()
        generator.manual_seed(int(seed if seed is not None else 0))
    return DataLoader(
        dataset=dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=num_workers,
        pin_memory=torch.cuda.is_available(),
        generator=generator,
        drop_last=False,
    )


def prepare_protocol_loaders_from_base(train_dataset, val_dataset, test_dataset, cfg: Dict, protocol: str, seed: int):
    batch_size = int(cfg.get('batch_size', 64))
    num_workers = int(cfg.get('num_workers', 4))

    few_shot_indices = None
    downstream_dataset = train_dataset
    ratio = parse_few_shot_ratio(protocol)
    if ratio is not None:
        n_select = max(1, int(round(len(train_dataset) * ratio)))
        rng = np.random.default_rng(seed + 2026)
        few_shot_indices = sorted(rng.choice(len(train_dataset), size=n_select, replace=False).tolist())
        downstream_dataset = Subset(train_dataset, few_shot_indices)

    train_loader = build_loader(downstream_dataset, batch_size=batch_size, shuffle=True, num_workers=num_workers,
                                seed=seed + 23)
    val_loader = build_loader(val_dataset, batch_size=batch_size, shuffle=False, num_workers=num_workers)
    test_loader = build_loader(test_dataset, batch_size=batch_size, shuffle=False, num_workers=num_workers)
    return train_loader, val_loader, test_loader, few_shot_indices


# ==========================================


class ProjectionHead(nn.Module):
    def __init__(self, in_dim: int, out_dim: int = 128):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(in_dim, in_dim),
            nn.BatchNorm1d(in_dim),
            nn.ReLU(),
            nn.Linear(in_dim, out_dim),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


class LeadAwareInputAdapter(nn.Module):
    def __init__(self, num_leads: int = 12, emb_dim: int = 16, dropout: float = 0.0):
        super().__init__()
        self.num_leads = num_leads
        self.embedding = nn.Embedding(num_leads, emb_dim)
        self.to_scale = nn.Linear(emb_dim, 1)
        self.to_bias = nn.Linear(emb_dim, 1)
        self.dropout = nn.Dropout(dropout)
        nn.init.zeros_(self.to_scale.weight)
        nn.init.zeros_(self.to_scale.bias)
        nn.init.zeros_(self.to_bias.weight)
        nn.init.zeros_(self.to_bias.bias)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        c = x.shape[1]
        device = x.device
        lead_ids = torch.arange(c, device=device)
        emb = self.embedding(lead_ids)
        scale = self.to_scale(emb).view(1, c, 1)
        bias = self.to_bias(emb).view(1, c, 1)
        out = x * (1.0 + scale) + bias
        return self.dropout(out)


def add_ecg_noise(x: torch.Tensor, noise_std: float = 0.02) -> torch.Tensor:
    return x + torch.randn_like(x) * noise_std


def random_window_resample(x: torch.Tensor, lengths: List[int]) -> torch.Tensor:
    b, c, t = x.shape
    out = torch.empty_like(x)
    for i in range(b):
        L = int(random.choice(lengths))
        L = max(16, min(L, t))
        start = 0 if L == t else random.randint(0, t - L)
        crop = x[i:i + 1, :, start:start + L]
        out[i:i + 1] = F.interpolate(crop, size=t, mode='linear', align_corners=False)
    return out


def apply_dynamic_lead_mask(
        x: torch.Tensor,
        mask_prob: float,
        min_drop: int,
        max_drop: int,
) -> Tuple[torch.Tensor, torch.Tensor]:
    b, c, _ = x.shape
    keep_mask = torch.ones((b, c, 1), device=x.device, dtype=x.dtype)
    if mask_prob <= 0.0:
        return x, keep_mask
    for i in range(b):
        if random.random() < mask_prob:
            n_drop = random.randint(min_drop, min(max_drop, c))
            drop_idx = random.sample(range(c), k=n_drop)
            keep_mask[i, drop_idx, 0] = 0.0
    return x * keep_mask, keep_mask


def _to_numpy_matrix(x) -> np.ndarray:
    if isinstance(x, np.ndarray):
        return x.astype(np.float32, copy=True)
    if hasattr(x, 'detach'):
        return x.detach().cpu().numpy().astype(np.float32, copy=True)
    return np.asarray(x, dtype=np.float32).copy()


def _print_matrix_stats(name: str, x) -> None:
    arr = _to_numpy_matrix(x)
    print(
        f'[{name}] shape={arr.shape} '
        f'min={arr.min():.6f} max={arr.max():.6f} mean={arr.mean():.6f}'
    )


def calculate_hndr(probs, targets, class_names, csv_path=DEFAULT_HARD_PAIRS):
    if not os.path.exists(csv_path):
        return None, None
    pairs_df = pd.read_csv(csv_path)
    pair_metrics = {}
    total_hard_samples = 0
    correct_hard_samples = 0
    for _, row in pairs_df.iterrows():
        disease_a = str(row['disease_a']).strip()
        disease_b = str(row['disease_b']).strip()
        if disease_a not in class_names or disease_b not in class_names:
            continue
        idx_a = class_names.index(disease_a)
        idx_b = class_names.index(disease_b)
        condition_a = (targets[:, idx_a] >= 0.5) & (targets[:, idx_b] < 0.5)
        condition_b = (targets[:, idx_a] < 0.5) & (targets[:, idx_b] >= 0.5)
        valid_mask = condition_a | condition_b
        subset_targets = targets[valid_mask]
        subset_probs = probs[valid_mask]
        if len(subset_targets) == 0:
            continue
        binary_targets = (subset_targets[:, idx_a] >= 0.5).astype(int)
        binary_preds = (subset_probs[:, idx_a] > subset_probs[:, idx_b]).astype(int)
        acc = accuracy_score(binary_targets, binary_preds)
        pair_metrics[f'{disease_a}_vs_{disease_b}'] = acc
        total_hard_samples += len(binary_targets)
        correct_hard_samples += np.sum(binary_preds == binary_targets)
    if not pair_metrics:
        return None, None
    hndr_pair = np.mean(list(pair_metrics.values()))
    hndr_inst = correct_hard_samples / total_hard_samples if total_hard_samples > 0 else 0.0
    return hndr_pair, hndr_inst


def tune_thresholds_per_class(probs: np.ndarray, targets: np.ndarray) -> np.ndarray:
    grid = np.arange(0.1, 0.91, 0.05)
    thresholds = np.full(probs.shape[1], 0.5, dtype=np.float32)
    for c in range(probs.shape[1]):
        best_thr, best_f1 = 0.5, -1.0
        y_true = targets[:, c].astype(int)
        for thr in grid:
            y_pred = (probs[:, c] >= thr).astype(int)
            score = f1_score(y_true, y_pred, zero_division=0)
            if score > best_f1:
                best_f1, best_thr = score, float(thr)
        thresholds[c] = best_thr
    return thresholds


def evaluate_from_probs(probs: np.ndarray, targets: np.ndarray, thresholds: Optional[np.ndarray] = None):
    if thresholds is None:
        thresholds = np.full(probs.shape[1], 0.5, dtype=np.float32)
    preds = (probs >= thresholds.reshape(1, -1)).astype(int)
    macro_auc = roc_auc_score(targets, probs, average='macro')
    auprc = average_precision_score(targets, probs, average='macro')
    macro_f1 = f1_score(targets, preds, average='macro', zero_division=0)
    idx_mi = CLASS_NAMES.index('MI')
    mi_f1 = f1_score(targets[:, idx_mi], preds[:, idx_mi], zero_division=0)
    hndr_pair, hndr_inst = calculate_hndr(probs, targets, CLASS_NAMES)
    return {
        'Macro_AUC': float(macro_auc),
        'AUPRC': float(auprc),
        'Macro_F1': float(macro_f1),
        'MI_F1': float(mi_f1),
        'HNDR_Pair': float(0.0 if hndr_pair is None else hndr_pair),
        'HNDR_Inst': float(0.0 if hndr_inst is None else hndr_inst),
    }


@torch.no_grad()
def collect_probs(model: nn.Module, loader, device: torch.device) -> Tuple[np.ndarray, np.ndarray]:
    model.eval()
    all_probs, all_targets = [], []
    for imgs, labels, _ in loader:
        imgs = imgs.to(device, non_blocking=True)
        logits = model.forward_cls(imgs)
        all_probs.append(torch.sigmoid(logits).cpu().numpy())
        all_targets.append(labels.numpy())
    return np.vstack(all_probs), np.vstack(all_targets)


# ==========================================
# 升级为支持 Subset (Few-Shot) 的 pos_weight 计算
# ==========================================
def compute_pos_weight(train_loader, device: torch.device) -> torch.Tensor:
    dataset = train_loader.dataset
    if isinstance(dataset, Subset) and hasattr(dataset.dataset, 'y_mh'):
        labels = np.asarray(dataset.dataset.y_mh)[np.asarray(dataset.indices)]
    elif hasattr(dataset, 'y_mh'):
        labels = np.asarray(dataset.y_mh)
    else:
        ys = []
        for _, batch_labels, _ in train_loader:
            ys.append(batch_labels.numpy())
        labels = np.vstack(ys)
    y = torch.tensor(labels, dtype=torch.float32)
    pos = y.sum(dim=0)
    neg = y.shape[0] - pos
    pos_weight = (neg / (pos + 1e-6)).clamp(min=1.0, max=20.0)
    return pos_weight.to(device)


def build_optimizer(model: nn.Module, cfg: Dict, protocol: str):
    if protocol == 'Linear_Probing':
        params = list(model.cls_head.parameters())
        if hasattr(model, 'lead_adapter') and model.lead_adapter is not None:
            # freeze lead adapter during LP to make evaluation stricter
            for p in model.lead_adapter.parameters():
                p.requires_grad = False
        return optim.AdamW(params, lr=cfg['lp_lr'], weight_decay=cfg['lp_weight_decay'])

    param_groups = [
        {'params': model.backbone.parameters(), 'lr': cfg['ft_lr'] * cfg['backbone_lr_mult'],
         'weight_decay': cfg['ft_weight_decay']},
        {'params': model.cls_head.parameters(), 'lr': cfg['ft_lr'] * cfg['head_lr_mult'],
         'weight_decay': cfg['ft_weight_decay']},
    ]
    if hasattr(model, 'lead_adapter') and model.lead_adapter is not None:
        param_groups.append(
            {'params': model.lead_adapter.parameters(), 'lr': cfg['ft_lr'] * cfg['head_lr_mult'],
             'weight_decay': cfg['ft_weight_decay']}
        )
    return optim.AdamW(param_groups)


def build_scheduler(optimizer, total_epochs: int):
    return optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=max(total_epochs, 1), eta_min=1e-6)


def reinit_head(model: nn.Module):
    if hasattr(model, 'cls_head') and isinstance(model.cls_head, nn.Linear):
        model.cls_head.reset_parameters()


def set_backbone_trainable(model: nn.Module, trainable: bool):
    for p in model.backbone.parameters():
        p.requires_grad = trainable
    if hasattr(model, 'lead_adapter') and model.lead_adapter is not None:
        for p in model.lead_adapter.parameters():
            p.requires_grad = trainable


def weak_view(x: torch.Tensor, cfg: Dict) -> torch.Tensor:
    x1 = add_ecg_noise(x, cfg['noise_std'])
    if cfg.get('use_polywindow', False):
        return random_window_resample(x1, cfg['polywindow_lengths'])
    return add_ecg_noise(x1, cfg['noise_std'])


def build_local_window_starts(total_len: int, window_size: int, num_windows: int, jitter: int) -> List[int]:
    if window_size >= total_len:
        return [0 for _ in range(num_windows)]
    if num_windows <= 1:
        return [max(0, (total_len - window_size) // 2)]
    step = max(1, (total_len - window_size) // (num_windows - 1))
    starts = []
    for i in range(num_windows):
        base = min(i * step, total_len - window_size)
        if jitter > 0:
            base += random.randint(-jitter, jitter)
        base = max(0, min(base, total_len - window_size))
        starts.append(int(base))
    starts.sort()
    return starts


def extract_local_windows(
        x: torch.Tensor,
        window_size: int,
        num_windows: int,
        jitter: int,
        shared_starts: Optional[List[List[int]]] = None,
) -> Tuple[torch.Tensor, List[List[int]]]:
    b, c, t = x.shape
    starts_per_sample = []
    chunks = []
    for i in range(b):
        starts = shared_starts[i] if shared_starts is not None else build_local_window_starts(t, window_size,
                                                                                              num_windows, jitter)
        starts_per_sample.append(starts)
        local_parts = []
        for s in starts:
            crop = x[i:i + 1, :, s:s + window_size]
            if crop.shape[-1] != window_size:
                crop = F.interpolate(crop, size=window_size, mode='linear', align_corners=False)
            local_parts.append(crop)
        chunks.append(torch.cat(local_parts, dim=0))
    return torch.cat(chunks, dim=0), starts_per_sample


def cosine_align_loss(z_a: torch.Tensor, z_b: torch.Tensor) -> torch.Tensor:
    z_a = F.normalize(z_a, dim=1)
    z_b = F.normalize(z_b, dim=1)
    return 1.0 - (z_a * z_b).sum(dim=1).mean()


def infer_backbone_feat_dim(model: nn.Module, device: torch.device) -> int:
    with torch.no_grad():
        dummy = torch.randn(1, 12, 1000, device=device)
        feat = model.backbone(dummy)
        feat = feat[0] if isinstance(feat, tuple) else feat
        return int(feat.shape[1])


def forward_backbone_features(model: nn.Module, x: torch.Tensor) -> torch.Tensor:
    if hasattr(model, 'lead_adapter') and model.lead_adapter is not None:
        x = model.lead_adapter(x)
    feat = model.backbone(x)
    feat = feat[0] if isinstance(feat, tuple) else feat
    return feat


def pretrain_with_leadaware_multiscale_relation(
        model: nn.Module,
        train_loader,
        cfg: Dict,
        device: torch.device,
        relation_matrix: np.ndarray,
        epochs: int,
        rhythm_projector: Optional[nn.Module] = None,
        local_projector: Optional[nn.Module] = None,
        stage_desc: str = 'MCKI Stage8 LeadAwareMultiScale',
):
    if epochs <= 0:
        return model, rhythm_projector, local_projector

    feat_dim = infer_backbone_feat_dim(model, device)
    if rhythm_projector is None:
        rhythm_projector = ProjectionHead(feat_dim, cfg['proj_dim']).to(device)
    if local_projector is None:
        local_projector = ProjectionHead(feat_dim, cfg['proj_dim']).to(device)

    loss_fn = MCKILossPro(
        alpha=cfg['alpha'],
        temperature=cfg['temperature'],
        hard_negative_threshold=cfg['hard_negative_threshold'],
        use_continuous_weights=cfg['use_continuous_weights'],
        relation_matrix=relation_matrix,
    ).to(device)

    params = list(model.backbone.parameters()) + list(rhythm_projector.parameters()) + list(
        local_projector.parameters())
    if hasattr(model, 'lead_adapter') and model.lead_adapter is not None:
        params += list(model.lead_adapter.parameters())
    optimizer = optim.AdamW(params, lr=cfg['pretrain_lr'], weight_decay=cfg['pretrain_weight_decay'])
    scheduler = build_scheduler(optimizer, epochs)
    log_every = max(1, int(cfg.get('log_pretrain_every', 5)))

    for epoch in tqdm(range(epochs), desc=stage_desc, leave=False):
        model.train()
        rhythm_projector.train()
        local_projector.train()
        if hasattr(model, 'lead_adapter') and model.lead_adapter is not None:
            model.lead_adapter.train()

        meter_total, meter_rhythm, meter_local, meter_align = [], [], [], []
        meter_masked_leads = []

        for imgs, labels_mh, _ in train_loader:
            imgs = imgs.to(device, non_blocking=True)
            labels_mh = labels_mh.to(device, non_blocking=True)

            view1 = add_ecg_noise(imgs, cfg['noise_std'])
            view2 = weak_view(imgs, cfg)

            if cfg.get('use_dynamic_lead_mask', False):
                view1, keep1 = apply_dynamic_lead_mask(
                    view1,
                    mask_prob=float(cfg.get('lead_mask_prob', 0.90)),
                    min_drop=int(cfg.get('lead_mask_min_drop', 1)),
                    max_drop=int(cfg.get('lead_mask_max_drop', 3)),
                )
                view2, keep2 = apply_dynamic_lead_mask(
                    view2,
                    mask_prob=float(cfg.get('lead_mask_prob', 0.90)),
                    min_drop=int(cfg.get('lead_mask_min_drop', 1)),
                    max_drop=int(cfg.get('lead_mask_max_drop', 3)),
                )
                meter_masked_leads.append(float((12.0 - keep1.squeeze(-1).sum(dim=1)).mean().item()))
                meter_masked_leads.append(float((12.0 - keep2.squeeze(-1).sum(dim=1)).mean().item()))

            feat1 = forward_backbone_features(model, view1)
            feat2 = forward_backbone_features(model, view2)
            z_rhythm_1 = rhythm_projector(feat1)
            z_rhythm_2 = rhythm_projector(feat2)
            loss_rhythm = loss_fn(z_rhythm_1, labels_mh, z_rhythm_2, labels_mh)

            loss_local = torch.tensor(0.0, device=device)
            loss_align = torch.tensor(0.0, device=device)

            if cfg.get('use_multiscale_local_branch', False):
                local_w = int(cfg['local_window_size'])
                local_k = int(cfg['local_num_windows'])
                local_jitter = int(cfg['local_jitter'])

                local_view1, starts_per_sample = extract_local_windows(view1, local_w, local_k, local_jitter,
                                                                       shared_starts=None)
                local_view2, _ = extract_local_windows(view2, local_w, local_k, local_jitter,
                                                       shared_starts=starts_per_sample)

                local_feat1 = forward_backbone_features(model, local_view1)
                local_feat2 = forward_backbone_features(model, local_view2)
                local_proj1 = local_projector(local_feat1)
                local_proj2 = local_projector(local_feat2)

                bsz = imgs.shape[0]
                local_proj1 = local_proj1.view(bsz, local_k, -1).mean(dim=1)
                local_proj2 = local_proj2.view(bsz, local_k, -1).mean(dim=1)

                loss_local = loss_fn(local_proj1, labels_mh, local_proj2, labels_mh)
                align_1 = cosine_align_loss(z_rhythm_1, local_proj1)
                align_2 = cosine_align_loss(z_rhythm_2, local_proj2)
                loss_align = 0.5 * (align_1 + align_2)

            loss = loss_rhythm + cfg.get('local_loss_weight', 0.30) * loss_local + cfg.get('align_loss_weight',
                                                                                           0.20) * loss_align

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            meter_total.append(float(loss.item()))
            meter_rhythm.append(float(loss_rhythm.item()))
            meter_local.append(float(loss_local.item()))
            meter_align.append(float(loss_align.item()))

        scheduler.step()

        if epoch == 0 or epoch == epochs - 1 or epoch % log_every == 0:
            masked_mean = 0.0 if len(meter_masked_leads) == 0 else float(np.mean(meter_masked_leads))
            print(
                f'[{stage_desc}] epoch={epoch + 1}/{epochs} '
                f'total={np.mean(meter_total):.4f} '
                f'rhythm={np.mean(meter_rhythm):.4f} '
                f'local={np.mean(meter_local):.4f} '
                f'align={np.mean(meter_align):.4f} '
                f'masked_leads={masked_mean:.2f}'
            )

    return model, rhythm_projector, local_projector


def bootstrap_confusion_matrix(model: nn.Module, train_loader, val_loader, cfg: Dict,
                               device: torch.device) -> np.ndarray:
    tmp_model = copy.deepcopy(model).to(device)
    reinit_head(tmp_model)
    set_backbone_trainable(tmp_model, False)
    optimizer = optim.AdamW(tmp_model.cls_head.parameters(), lr=cfg['bootstrap_lp_lr'],
                            weight_decay=cfg['bootstrap_lp_weight_decay'])
    pos_weight = compute_pos_weight(train_loader, device)
    criterion = nn.BCEWithLogitsLoss(pos_weight=pos_weight)

    best_state = None
    best_score = -1.0
    for _ in tqdm(range(cfg['bootstrap_lp_epochs']), desc='Bootstrap LP', leave=False):
        tmp_model.train()
        for imgs, labels, _ in train_loader:
            imgs = imgs.to(device, non_blocking=True)
            labels = labels.to(device, non_blocking=True)
            optimizer.zero_grad()
            logits = tmp_model.forward_cls(imgs)
            loss = criterion(logits, labels)
            loss.backward()
            optimizer.step()
        val_probs, val_targets = collect_probs(tmp_model, val_loader, device)
        val_metrics = evaluate_from_probs(val_probs, val_targets)
        score = val_metrics['AUPRC']
        if score > best_score:
            best_score = score
            best_state = copy.deepcopy(tmp_model.state_dict())

    if best_state is not None:
        tmp_model.load_state_dict(best_state)
    probs, targets = collect_probs(tmp_model, val_loader, device)
    return estimate_confusion_matrix_from_probs(probs, targets)


def train_protocol(model: nn.Module, train_loader, val_loader, cfg: Dict, device: torch.device, protocol: str):
    reinit_head(model)
    if protocol == 'Linear_Probing':
        set_backbone_trainable(model, False)
        epochs = cfg['lp_epochs']
    else:
        set_backbone_trainable(model, True)
        epochs = cfg['finetune_epochs']

    optimizer = build_optimizer(model, cfg, protocol)
    scheduler = build_scheduler(optimizer, epochs)
    criterion = nn.BCEWithLogitsLoss(pos_weight=compute_pos_weight(train_loader, device))

    best_state = None
    best_score = -1.0
    best_thresholds = np.full(len(CLASS_NAMES), 0.5, dtype=np.float32)
    patience = 0

    for _ in tqdm(range(epochs), desc=f'{protocol}', leave=False):
        model.train()
        for imgs, labels, _ in train_loader:
            imgs = imgs.to(device, non_blocking=True)
            labels = labels.to(device, non_blocking=True)
            optimizer.zero_grad()
            logits = model.forward_cls(imgs)
            loss = criterion(logits, labels)
            loss.backward()
            optimizer.step()
        scheduler.step()

        val_probs, val_targets = collect_probs(model, val_loader, device)
        thresholds = tune_thresholds_per_class(val_probs, val_targets)
        val_metrics = evaluate_from_probs(val_probs, val_targets, thresholds)
        score = float(val_metrics[cfg['monitor_metric']])
        if score > best_score:
            best_score = score
            best_state = copy.deepcopy(model.state_dict())
            best_thresholds = thresholds.copy()
            patience = 0
        else:
            patience += 1
        if patience >= cfg['early_stop_patience']:
            break

    if best_state is not None:
        model.load_state_dict(best_state)
    return model, best_thresholds


def attach_lead_adapter_if_needed(model: nn.Module, cfg: Dict):
    if cfg.get('use_lead_aware_input', False):
        model.lead_adapter = LeadAwareInputAdapter(
            num_leads=12,
            emb_dim=int(cfg.get('lead_emb_dim', 16)),
            dropout=float(cfg.get('lead_adapter_dropout', 0.0)),
        )
    else:
        model.lead_adapter = None
    return model


def wrap_forward_cls_for_leadaware(model: nn.Module):
    if not hasattr(model, '_original_forward_cls'):
        model._original_forward_cls = model.forward_cls

    def _forward_cls_leadaware(x: torch.Tensor):
        if hasattr(model, 'lead_adapter') and model.lead_adapter is not None:
            x = model.lead_adapter(x)
        return model._original_forward_cls(x)

    model.forward_cls = _forward_cls_leadaware
    return model


def build_model(cfg: Dict, device: torch.device):
    model, _ = build_MCKI_backbone(cfg['backbone_name'], num_classes=len(CLASS_NAMES), device=device, cfg=cfg)
    model = attach_lead_adapter_if_needed(model, cfg)
    model = wrap_forward_cls_for_leadaware(model)
    model = model.to(device)
    return model


# ==========================================
# 更新的 5 协议调度版 run_single_seed
# ==========================================
def run_single_seed(seed: int, cfg: Dict, save_dir: str) -> List[Dict]:
    seed_everything(seed)
    ensure_dir(save_dir)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    # 提取完整的数据集和 Loader，作为提取 subset 的 Base
    base_train_loader, base_val_loader, base_test_loader = get_dataloader_v3(
        DEFAULT_DATA_DIR,
        batch_size=cfg['batch_size'],
        num_workers=cfg['num_workers'],
    )
    train_dataset = base_train_loader.dataset
    val_dataset = base_val_loader.dataset
    test_dataset = base_test_loader.dataset

    model = build_model(cfg, device)

    prior = load_prior_matrix(cfg.get('relation_matrix_values'))
    warmup_epochs = min(cfg['warmup_pretrain_epochs'], cfg['pretrain_epochs'])
    hybrid_epochs = max(0, cfg['pretrain_epochs'] - warmup_epochs)

    print(
        f'\n🚀 [Seed {seed}] Stage8-LeadAware-MultiScale-MCKI (5 Protocols) | warmup={warmup_epochs} | hybrid={hybrid_epochs}')
    _print_matrix_stats('S_prior', prior)

    # 预训练阶段（使用全局全量数据 base_train_loader）
    model, rhythm_projector, local_projector = pretrain_with_leadaware_multiscale_relation(
        model=model,
        train_loader=base_train_loader,
        cfg=cfg,
        device=device,
        relation_matrix=prior,
        epochs=warmup_epochs,
        rhythm_projector=None,
        local_projector=None,
        stage_desc='MCKI Stage8 Warmup(prior+leadaware+multiscale)',
    )

    conf = bootstrap_confusion_matrix(model, base_train_loader, base_val_loader, cfg, device)
    _print_matrix_stats('S_conf', conf)

    hybrid = blend_relation_matrices(prior, conf, cfg['lambda_prior'], cfg['lambda_conf'])
    _print_matrix_stats('S_hybrid_lambda05', hybrid)

    seed_dir = os.path.join(save_dir, f'seed_{seed}')
    ensure_dir(seed_dir)
    save_relation_artifacts(seed_dir, prior, conf, hybrid)
    with open(os.path.join(seed_dir, 'leadaware_multiscale_config.json'), 'w', encoding='utf-8') as f:
        json.dump({
            'lambda_prior': float(cfg['lambda_prior']),
            'lambda_conf': float(cfg['lambda_conf']),
            'use_multiscale_local_branch': bool(cfg['use_multiscale_local_branch']),
            'local_window_size': int(cfg['local_window_size']),
            'local_num_windows': int(cfg['local_num_windows']),
            'local_jitter': int(cfg['local_jitter']),
            'local_loss_weight': float(cfg['local_loss_weight']),
            'align_loss_weight': float(cfg['align_loss_weight']),
            'use_lead_aware_input': bool(cfg['use_lead_aware_input']),
            'lead_emb_dim': int(cfg['lead_emb_dim']),
            'use_dynamic_lead_mask': bool(cfg['use_dynamic_lead_mask']),
            'lead_mask_prob': float(cfg['lead_mask_prob']),
            'lead_mask_min_drop': int(cfg['lead_mask_min_drop']),
            'lead_mask_max_drop': int(cfg['lead_mask_max_drop']),
            'warmup_epochs': int(warmup_epochs),
            'hybrid_epochs': int(hybrid_epochs),
        }, f, ensure_ascii=False, indent=2)

    if hybrid_epochs > 0:
        model, rhythm_projector, local_projector = pretrain_with_leadaware_multiscale_relation(
            model=model,
            train_loader=base_train_loader,
            cfg=cfg,
            device=device,
            relation_matrix=hybrid,
            epochs=hybrid_epochs,
            rhythm_projector=rhythm_projector,
            local_projector=local_projector,
            stage_desc='MCKI Stage8 Hybrid(lambda05+leadaware+multiscale)',
        )

    rows = []
    pretrained_state = copy.deepcopy(model.state_dict())

    # 按照 5 个协议循环微调和测试
    for protocol in DEFAULT_PROTOCOLS:
        proto_train_loader, proto_val_loader, proto_test_loader, few_shot_indices = prepare_protocol_loaders_from_base(
            train_dataset, val_dataset, test_dataset, cfg, protocol, seed
        )

        proto_model = build_model(cfg, device)
        proto_model.load_state_dict(pretrained_state)
        proto_model, thresholds = train_protocol(proto_model, proto_train_loader, proto_val_loader, cfg, device,
                                                 protocol)
        val_probs, val_targets = collect_probs(proto_model, proto_val_loader, device)
        test_probs, test_targets = collect_probs(proto_model, proto_test_loader, device)
        val_metrics = evaluate_from_probs(val_probs, val_targets, thresholds)
        test_metrics = evaluate_from_probs(test_probs, test_targets, thresholds)

        row = {
            'seed': seed,
            'protocol': protocol,
            'n_train_samples': len(proto_train_loader.dataset),
            'few_shot_indices': json.dumps(few_shot_indices,
                                           ensure_ascii=False) if few_shot_indices is not None else '',
            **{f'val_{k}': v for k, v in val_metrics.items()},
            **{f'test_{k}': v for k, v in test_metrics.items()},
        }
        rows.append(row)
    return rows


def summarize_and_save(all_rows: List[Dict], save_dir: str):
    ensure_dir(save_dir)
    df = pd.DataFrame(all_rows)
    df.to_csv(os.path.join(save_dir, 'per_seed_results.csv'), index=False)
    summary = {}
    for protocol in DEFAULT_PROTOCOLS:
        sub = df[df['protocol'] == protocol]
        metric_cols = ['test_Macro_AUC', 'test_AUPRC', 'test_Macro_F1', 'test_MI_F1', 'test_HNDR_Pair',
                       'test_HNDR_Inst']
        mean = sub[metric_cols].mean()
        std = sub[metric_cols].std()
        summary[protocol] = {
            'Macro_AUC': f"{mean['test_Macro_AUC']:.4f} ± {std['test_Macro_AUC']:.4f}",
            'AUPRC': f"{mean['test_AUPRC']:.4f} ± {std['test_AUPRC']:.4f}",
            'F1': f"{mean['test_Macro_F1']:.4f} ± {std['test_Macro_F1']:.4f}",
            'MI_F1': f"{mean['test_MI_F1']:.4f} ± {std['test_MI_F1']:.4f}",
            'HNDR_Pair': f"{mean['test_HNDR_Pair']:.4f} ± {std['test_HNDR_Pair']:.4f}",
            'HNDR_Inst': f"{mean['test_HNDR_Inst']:.4f} ± {std['test_HNDR_Inst']:.4f}",
        }
        pd.DataFrame([summary[protocol]], index=['MCKI_Pro']).to_csv(
            os.path.join(save_dir, f'master_table_{protocol}.csv'))
    with open(os.path.join(save_dir, 'summary.json'), 'w', encoding='utf-8') as f:
        json.dump(summary, f, ensure_ascii=False, indent=2)


def main():
    ensure_dir(DEFAULT_SAVE_DIR)
    print('\n' + '▼' * 80)
    print('🔬 启动 Stage8 (5 Protocols): Lead-aware MultiScale MCKI')
    print(f'   Protocols = {DEFAULT_PROTOCOLS}')
    print(f'   Seeds     = {DEFAULT_SEEDS}')
    print(json.dumps(CFG, indent=2, ensure_ascii=False))
    print('▲' * 80)

    all_rows = []
    for seed in DEFAULT_SEEDS:
        rows = run_single_seed(seed, CFG, DEFAULT_SAVE_DIR)
        all_rows.extend(rows)
        pd.DataFrame(all_rows).to_csv(os.path.join(DEFAULT_SAVE_DIR, 'per_seed_results.csv'), index=False)

    summarize_and_save(all_rows, DEFAULT_SAVE_DIR)
    print(f'✅ Saved to {DEFAULT_SAVE_DIR}')


if __name__ == '__main__':
    main()