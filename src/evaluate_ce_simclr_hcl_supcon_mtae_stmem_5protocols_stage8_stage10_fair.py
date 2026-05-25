import os
import sys
import json
import copy
import random
import argparse
from typing import Dict, List, Tuple, Optional

import numpy as np
import pandas as pd
from tqdm import tqdm

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Subset
from sklearn.metrics import f1_score, roc_auc_score, average_precision_score, accuracy_score

# ==============================================================================
# 0. 路径与导入
# ==============================================================================
CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.abspath(os.path.join(CURRENT_DIR, '.'))
if CURRENT_DIR not in sys.path:
    sys.path.insert(0, CURRENT_DIR)

ENCODER_PATH = os.path.join(CURRENT_DIR, 'models', 'encoder')
if os.path.exists(ENCODER_PATH) and ENCODER_PATH not in sys.path:
    sys.path.insert(0, ENCODER_PATH)

from dataset_v3 import get_dataloader_v3
from train_v3_2stage import TwoStageModel
from MCKI_loss_pro import MCKILossPro
from st_mem import st_mem_vit_small_dec256d4b
from models.encoder.st_mem_vit import st_mem_vit_small

# ==============================================================================
# 1. 常量与配置
# ==============================================================================
CLASS_NAMES = ['NORM', 'MI', 'STTC', 'CD', 'HYP']
DEFAULT_DATA_DIR = './data/processed_v3'
DEFAULT_SAVE_DIR = './formal_eval_outputs_ssl_stmem_stage8_stage10_fair'
DEFAULT_PROTOCOLS = ['Full_Finetune', 'Linear_Probing', 'Few_Shot_1%', 'Few_Shot_10%', 'Few_Shot_25%']
DEFAULT_SEEDS = [42, 123, 1024]
DEFAULT_MODELS = ['CE', 'SimCLR', 'HCL', 'SupCon', 'MTAE', 'ST_MEM']

# 尽量向 Stage8/10 下游壳对齐；不同方法保留必要的一阶段预训练差异
COMMON_FAIR_SHELL = {
    'batch_size': 64,
    'pretrain_epochs': 40,
    'finetune_epochs': 40,
    'lp_epochs': 25,
    'freeze_backbone_epochs': 0,
    'ft_lr': 1e-4,
    'ft_weight_decay': 1e-5,
    'backbone_lr_mult': 0.1,
    'head_lr_mult': 1.0,
    'lp_lr': 1e-3,
    'lp_weight_decay': 1e-4,
    'few_shot_ft_lr': 1e-4,
    'scheduler': 'cosine',
    'min_lr': 1e-6,
    'early_stop_patience': 8,
    'monitor_metric': 'AUPRC',
    'tune_thresholds': True,
    'num_workers': 4,
    'grad_clip': 0.0,
    'use_pos_weight': True,
    'proj_dim': 128,
    'noise_std': 0.02,
}

MODEL_CONFIGS: Dict[str, Dict] = {
    'CE': {
        **COMMON_FAIR_SHELL,
        'batch_size': 64,
        'ft_lr': 1e-4,
        'ft_weight_decay': 1e-5,
        'backbone_lr_mult': 1.0,  # 从头监督时不区分 backbone/head
        'head_lr_mult': 1.0,
    },
    'SimCLR': {
        **COMMON_FAIR_SHELL,
        'pretrain_lr': 3e-4,
        'pretrain_weight_decay': 1e-4,
        'temperature': 0.07,
    },
    'HCL': {
        **COMMON_FAIR_SHELL,
        'pretrain_lr': 3e-4,
        'pretrain_weight_decay': 1e-4,
        'temperature': 0.07,
        'beta': 1.0,
    },
    'SupCon': {
        **COMMON_FAIR_SHELL,
        'pretrain_lr': 3e-4,
        'pretrain_weight_decay': 1e-4,
        'temperature': 0.07,
    },
    'MTAE': {
        **COMMON_FAIR_SHELL,
        'pretrain_lr': 1e-3,
        'pretrain_weight_decay': 1e-4,
        'mask_ratio': 0.4,
    },
    'ST_MEM': {
        **COMMON_FAIR_SHELL,
        'batch_size': 32,  # ST-MEM 较吃显存，必要时可命令行覆盖
        'pretrain_lr': 1e-3,
        'pretrain_weight_decay': 5e-2,
        'mask_ratio': 0.75,
        'seq_len': 1000,
        'patch_size': 50,
        'num_leads': 12,
    },
}

# ==============================================================================
# 2. 基础工具
# ==============================================================================
def seed_everything(seed: int):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def ensure_dir(path: str):
    os.makedirs(path, exist_ok=True)


def resolve_confusable_csv_path(csv_path: Optional[str] = None) -> str:
    candidates = []
    if csv_path:
        candidates.append(csv_path)
        if not os.path.isabs(csv_path):
            candidates.append(os.path.join(PROJECT_ROOT, csv_path))
    candidates.extend([
        os.path.join(PROJECT_ROOT, 'src3', 'confusable_pairs_v1.csv'),
        os.path.join(PROJECT_ROOT, 'confusable_pairs_v1.csv'),
        './confusable_pairs_v1.csv',
    ])
    for p in candidates:
        if p and os.path.exists(p):
            return p
    raise FileNotFoundError(f'未找到 confusable_pairs_v1.csv，候选路径: {candidates}')


def parse_few_shot_ratio(protocol: str) -> Optional[float]:
    if not protocol.startswith('Few_Shot_'):
        return None
    suffix = protocol[len('Few_Shot_'):].strip()
    if not suffix.endswith('%'):
        raise ValueError(f'无法解析 few-shot 协议比例: {protocol}')
    ratio = float(suffix[:-1]) / 100.0
    if not (0.0 < ratio <= 1.0):
        raise ValueError(f'few-shot 比例非法: {ratio}')
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


def prepare_protocol_loaders(data_dir: str, cfg: Dict, protocol: str, seed: int):
    batch_size = int(cfg.get('batch_size', 64))
    num_workers = int(cfg.get('num_workers', 4))

    base_train_loader, base_val_loader, base_test_loader = get_dataloader_v3(
        data_dir,
        batch_size=batch_size,
        num_workers=num_workers,
    )
    train_dataset = base_train_loader.dataset
    val_dataset = base_val_loader.dataset
    test_dataset = base_test_loader.dataset

    few_shot_indices = None
    downstream_dataset = train_dataset
    few_ratio = parse_few_shot_ratio(protocol)
    if few_ratio is not None:
        n_select = max(1, int(round(len(train_dataset) * few_ratio)))
        rng = np.random.default_rng(seed + 2026)
        few_shot_indices = sorted(rng.choice(len(train_dataset), size=n_select, replace=False).tolist())
        downstream_dataset = Subset(train_dataset, few_shot_indices)

    train_loader = build_loader(downstream_dataset, batch_size, True, num_workers, seed + 23)
    val_loader = build_loader(val_dataset, batch_size, False, num_workers)
    test_loader = build_loader(test_dataset, batch_size, False, num_workers)
    pretrain_loader = build_loader(train_dataset, batch_size, True, num_workers, seed + 11)
    return pretrain_loader, train_loader, val_loader, test_loader, few_shot_indices


def batch_get_labels(batch):
    if isinstance(batch, dict):
        if 'label' in batch:
            return batch['label']
        raise KeyError('dict batch 中未找到 label')
    if isinstance(batch, (list, tuple)) and len(batch) >= 2:
        return batch[1]
    raise TypeError(f'无法从 batch 中提取 labels, type={type(batch)}')


def batch_get_model_input(batch):
    if isinstance(batch, dict):
        return batch
    if isinstance(batch, (list, tuple)):
        return batch[0]
    return batch


def model_forward_cls(model: nn.Module, x):
    if hasattr(model, 'forward_cls'):
        return model.forward_cls(x)
    return model(x)


def has_backbone_and_head(model: nn.Module) -> bool:
    return hasattr(model, 'backbone') and hasattr(model, 'cls_head')


# ==============================================================================
# 3. 损失函数与增强
# ==============================================================================
def add_ecg_noise(x: torch.Tensor, noise_std: float = 0.02) -> torch.Tensor:
    return x + torch.randn_like(x) * noise_std


class SupConLoss(nn.Module):
    def __init__(self, temperature=0.07):
        super().__init__()
        self.temperature = temperature

    def forward(self, z1, z2, labels):
        device = z1.device
        batch_size = z1.shape[0]
        features = torch.cat([z1, z2], dim=0)
        features = nn.functional.normalize(features, dim=1)

        labels = labels.contiguous().view(-1, 1)
        labels = torch.cat([labels, labels], dim=0)
        mask = torch.eq(labels, labels.T).float().to(device)

        logits = torch.matmul(features, features.T) / self.temperature
        logits_max, _ = torch.max(logits, dim=1, keepdim=True)
        logits = logits - logits_max.detach()

        logits_mask = torch.scatter(
            torch.ones_like(mask), 1, torch.arange(batch_size * 2, device=device).view(-1, 1), 0
        )
        mask = mask * logits_mask

        exp_logits = torch.exp(logits) * logits_mask
        log_prob = logits - torch.log(exp_logits.sum(1, keepdim=True) + 1e-8)
        mean_log_prob_pos = (mask * log_prob).sum(1) / (mask.sum(1) + 1e-8)
        return -mean_log_prob_pos.mean()


class SimCLRLoss(nn.Module):
    def __init__(self, temperature=0.07):
        super().__init__()
        self.temp = temperature
        self.criterion = nn.CrossEntropyLoss()

    def forward(self, z_i, z_j):
        z_i = nn.functional.normalize(z_i, dim=1)
        z_j = nn.functional.normalize(z_j, dim=1)
        logits = torch.matmul(z_i, z_j.T) / self.temp
        labels = torch.arange(logits.size(0), device=z_i.device)
        return self.criterion(logits, labels)


class HCLLoss(nn.Module):
    def __init__(self, temperature=0.07, beta=1.0):
        super().__init__()
        self.temp = temperature
        self.beta = beta

    def forward(self, z1, z2):
        z1 = nn.functional.normalize(z1, dim=1)
        z2 = nn.functional.normalize(z2, dim=1)
        batch_size = z1.size(0)

        sim_matrix = torch.exp(torch.matmul(z1, z2.T) / self.temp)
        sim_pos = torch.diag(sim_matrix)

        sim_matrix_scaled = torch.exp(torch.matmul(z1, z2.T) / self.temp * self.beta)
        mask = torch.eye(batch_size, dtype=torch.bool, device=z1.device)
        sim_matrix_scaled = sim_matrix_scaled.masked_fill(mask, 0.0)

        neg_sum = sim_matrix_scaled.sum(dim=1)
        loss = -torch.log(sim_pos / (sim_pos + neg_sum + 1e-8)).mean()
        return loss


class ProjectionHead(nn.Module):
    def __init__(self, in_dim: int, out_dim: int = 128):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(in_dim, in_dim),
            nn.BatchNorm1d(in_dim),
            nn.ReLU(),
            nn.Linear(in_dim, out_dim)
        )

    def forward(self, x):
        return self.net(x)


class MTAE_Pretrainer(nn.Module):
    def __init__(self, backbone, feature_dim, seq_len=1000, in_channels=12, mask_ratio=0.4):
        super().__init__()
        self.backbone = backbone
        self.mask_ratio = mask_ratio
        self.decoder = nn.Sequential(
            nn.Linear(feature_dim, 256),
            nn.ReLU(),
            nn.Linear(256, 512),
            nn.ReLU(),
            nn.Linear(512, in_channels * seq_len)
        )

    def forward(self, x):
        b, c, l = x.shape
        mask = (torch.rand(b, 1, l, device=x.device) < self.mask_ratio).expand_as(x)
        masked_x = x.clone()
        masked_x[mask] = 0.0
        feat = self.backbone(masked_x)
        if isinstance(feat, tuple):
            feat = feat[0]
        reconstructed = self.decoder(feat).view(b, c, l)
        return nn.functional.mse_loss(reconstructed[mask], x[mask])


class STMEMClassifierWrapper(nn.Module):
    def __init__(self, num_classes=5, seq_len=1000, patch_size=50, num_leads=12):
        super().__init__()
        self.model = st_mem_vit_small(
            num_leads=num_leads,
            num_classes=num_classes,
            seq_len=seq_len,
            patch_size=patch_size,
        )
        # 给统一 optimizer 壳暴露 backbone / cls_head
        self.backbone = self.model.encoder if hasattr(self.model, 'encoder') else self.model
        self.cls_head = self.model.head if hasattr(self.model, 'head') else getattr(self.model, 'fc_norm', self.model)

    def forward_cls(self, x: torch.Tensor) -> torch.Tensor:
        return self.model(x)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.model(x)


def transfer_stmem_weights(pretrain_model: nn.Module, classifier_model: STMEMClassifierWrapper):
    src_state = pretrain_model.encoder.state_dict() if hasattr(pretrain_model, 'encoder') else pretrain_model.state_dict()
    if hasattr(classifier_model.model, 'encoder'):
        load_info = classifier_model.model.encoder.load_state_dict(src_state, strict=False)
    else:
        load_info = classifier_model.model.load_state_dict(src_state, strict=False)
    print(
        f"   [ST_MEM] encoder weights transferred | "
        f"missing={len(getattr(load_info, 'missing_keys', []))} | "
        f"unexpected={len(getattr(load_info, 'unexpected_keys', []))}"
    )
    return classifier_model


# ==============================================================================
# 4. 模型构建 / 预训练
# ==============================================================================
def build_downstream_model(mode_name: str, device: torch.device, cfg: Dict, num_classes: int = 5):
    if mode_name == 'ST_MEM':
        model = STMEMClassifierWrapper(
            num_classes=num_classes,
            seq_len=int(cfg.get('seq_len', 1000)),
            patch_size=int(cfg.get('patch_size', 50)),
            num_leads=int(cfg.get('num_leads', 12)),
        ).to(device)
        backbone_dim = 384
        return model, backbone_dim

    model = TwoStageModel(num_classes=num_classes).to(device)
    with torch.no_grad():
        dummy_in = torch.randn(1, 12, 1000, device=device)
        dummy_out = model.backbone(dummy_in)
        backbone_dim = dummy_out[0].shape[1] if isinstance(dummy_out, tuple) else dummy_out.shape[1]
    return model, backbone_dim


def build_pretrain_model(mode_name: str, device: torch.device, cfg: Dict, num_classes: int = 5):
    if mode_name == 'ST_MEM':
        model = st_mem_vit_small_dec256d4b(
            seq_len=int(cfg.get('seq_len', 1000)),
            patch_size=int(cfg.get('patch_size', 50)),
            num_leads=int(cfg.get('num_leads', 12)),
        ).to(device)
        backbone_dim = 384
        return model, backbone_dim
    return build_downstream_model(mode_name, device, cfg, num_classes=num_classes)


def build_loss(mode_name: str, cfg: Dict):
    if mode_name in ['CE', 'MTAE', 'ST_MEM']:
        return None
    if mode_name == 'SimCLR':
        return SimCLRLoss(temperature=float(cfg.get('temperature', 0.07)))
    if mode_name == 'HCL':
        return HCLLoss(temperature=float(cfg.get('temperature', 0.07)), beta=float(cfg.get('beta', 1.0)))
    if mode_name == 'SupCon':
        return SupConLoss(temperature=float(cfg.get('temperature', 0.07)))
    raise ValueError(f'Unsupported mode_name: {mode_name}')


def pretrain_and_get_state(mode_name: str, model: nn.Module, backbone_dim: int, train_loader, cfg: Dict, device: torch.device):
    if mode_name == 'CE':
        print('   [提示] CE 跳过预训练。')
        return copy.deepcopy(model.state_dict())

    pretrain_epochs = int(cfg.get('pretrain_epochs', 40))
    pretrain_lr = float(cfg.get('pretrain_lr', 3e-4))
    pretrain_wd = float(cfg.get('pretrain_weight_decay', 1e-4))
    noise_std = float(cfg.get('noise_std', 0.02))

    if mode_name == 'ST_MEM':
        mask_ratio = float(cfg.get('mask_ratio', 0.75))
        optimizer = optim.AdamW(model.parameters(), lr=pretrain_lr, weight_decay=pretrain_wd)
        for _ in tqdm(range(pretrain_epochs), desc='ST_MEM Pre-train', leave=False):
            model.train()
            for imgs, _, _ in train_loader:
                imgs = imgs.to(device, non_blocking=True)
                optimizer.zero_grad()
                out = model(imgs, mask_ratio=mask_ratio)
                loss = out['loss']
                loss.backward()
                optimizer.step()
        classifier_model, _ = build_downstream_model('ST_MEM', device, cfg, num_classes=len(CLASS_NAMES))
        classifier_model = transfer_stmem_weights(model, classifier_model)
        return copy.deepcopy(classifier_model.state_dict())

    if mode_name == 'MTAE':
        mtae = MTAE_Pretrainer(model.backbone, backbone_dim, mask_ratio=float(cfg.get('mask_ratio', 0.4))).to(device)
        optimizer = optim.AdamW(mtae.parameters(), lr=pretrain_lr, weight_decay=pretrain_wd)
        for _ in tqdm(range(pretrain_epochs), desc='MTAE Pre-train', leave=False):
            mtae.train()
            for imgs, _, _ in train_loader:
                imgs = imgs.to(device, non_blocking=True)
                optimizer.zero_grad()
                loss = mtae(imgs)
                loss.backward()
                optimizer.step()
        return copy.deepcopy(model.state_dict())

    projector = ProjectionHead(backbone_dim, out_dim=int(cfg.get('proj_dim', 128))).to(device)
    loss_fn = build_loss(mode_name, cfg)
    params = list(model.backbone.parameters()) + list(projector.parameters())
    optimizer = optim.AdamW(params, lr=pretrain_lr, weight_decay=pretrain_wd)

    for _ in tqdm(range(pretrain_epochs), desc=f'{mode_name} Pre-train', leave=False):
        model.train()
        projector.train()
        for imgs, _, anchors in train_loader:
            imgs = imgs.to(device, non_blocking=True)
            anchors = anchors.to(device, non_blocking=True)
            optimizer.zero_grad()

            feat1 = model.backbone(imgs)
            if isinstance(feat1, tuple):
                feat1 = feat1[0]
            proj1 = projector(feat1)

            imgs_aug = add_ecg_noise(imgs, noise_std=noise_std)
            feat2 = model.backbone(imgs_aug)
            if isinstance(feat2, tuple):
                feat2 = feat2[0]
            proj2 = projector(feat2)

            if mode_name == 'SupCon':
                loss = loss_fn(proj1, proj2, anchors)
            else:
                loss = loss_fn(proj1, proj2)

            loss.backward()
            optimizer.step()

    return copy.deepcopy(model.state_dict())


# ==============================================================================
# 5. 评估工具
# ==============================================================================
def calculate_hndr(probs, targets, class_names, csv_path: str):
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
    hndr_pair = float(np.mean(list(pair_metrics.values())))
    hndr_inst = float(correct_hard_samples / total_hard_samples) if total_hard_samples > 0 else 0.0
    return hndr_pair, hndr_inst


def tune_thresholds_per_class(probs: np.ndarray, targets: np.ndarray, grid=None) -> np.ndarray:
    if grid is None:
        grid = np.arange(0.1, 0.91, 0.05)
    thresholds = np.full(probs.shape[1], 0.5, dtype=np.float32)
    for c in range(probs.shape[1]):
        best_thr, best_f1 = 0.5, -1.0
        y_true = targets[:, c].astype(int)
        for thr in grid:
            y_pred = (probs[:, c] >= thr).astype(int)
            score = f1_score(y_true, y_pred, zero_division=0)
            if score > best_f1:
                best_f1 = score
                best_thr = float(thr)
        thresholds[c] = best_thr
    return thresholds


def evaluate_from_probs(probs: np.ndarray, targets: np.ndarray, class_names: List[str], thresholds: Optional[np.ndarray], csv_path: str):
    if thresholds is None:
        thresholds = np.full(probs.shape[1], 0.5, dtype=np.float32)
    preds = (probs >= thresholds.reshape(1, -1)).astype(int)
    macro_auc = roc_auc_score(targets, probs, average='macro')
    auprc = average_precision_score(targets, probs, average='macro')
    macro_f1 = f1_score(targets, preds, average='macro', zero_division=0)
    mi_idx = class_names.index('MI')
    mi_f1 = f1_score(targets[:, mi_idx], preds[:, mi_idx], zero_division=0)
    hndr_pair, hndr_inst = calculate_hndr(probs, targets, class_names, csv_path)
    return {
        'Macro_AUC': float(macro_auc),
        'AUPRC': float(auprc),
        'Macro_F1': float(macro_f1),
        'MI_F1': float(mi_f1),
        'HNDR_Pair': float(0.0 if hndr_pair is None else hndr_pair),
        'HNDR_Inst': float(0.0 if hndr_inst is None else hndr_inst),
    }


def collect_probs(model: nn.Module, loader, device: torch.device):
    model.eval()
    all_probs, all_targets = [], []
    with torch.no_grad():
        for batch in loader:
            x = batch_get_model_input(batch)
            y = batch_get_labels(batch)
            if isinstance(x, dict):
                x = {k: v.to(device, non_blocking=True) for k, v in x.items()}
            else:
                x = x.to(device, non_blocking=True)
            logits = model_forward_cls(model, x)
            all_probs.append(torch.sigmoid(logits).cpu().numpy())
            all_targets.append(y.cpu().numpy() if torch.is_tensor(y) else np.asarray(y))
    return np.vstack(all_probs), np.vstack(all_targets)


# ==============================================================================
# 6. 下游训练
# ==============================================================================
def compute_pos_weight(train_loader, device: torch.device) -> torch.Tensor:
    dataset = train_loader.dataset
    if isinstance(dataset, Subset) and hasattr(dataset.dataset, 'y_mh'):
        labels = np.asarray(dataset.dataset.y_mh)[np.asarray(dataset.indices)]
    elif hasattr(dataset, 'y_mh'):
        labels = np.asarray(dataset.y_mh)
    else:
        ys = []
        for batch in train_loader:
            y = batch_get_labels(batch)
            ys.append(y.cpu().numpy() if torch.is_tensor(y) else np.asarray(y))
        labels = np.vstack(ys)
    y = torch.tensor(labels, dtype=torch.float32)
    pos = y.sum(dim=0)
    neg = y.shape[0] - pos
    pos_weight = (neg / (pos + 1e-6)).clamp(min=1.0, max=20.0)
    return pos_weight.to(device)


def get_linear_probe_modules(model: nn.Module) -> List[nn.Module]:
    candidates = []
    def _append_if_exists(obj, attr: str):
        if hasattr(obj, attr):
            module = getattr(obj, attr)
            if isinstance(module, nn.Module):
                candidates.append(module)
    _append_if_exists(model, 'cls_head')
    _append_if_exists(model, 'head')
    _append_if_exists(model, 'classifier')
    _append_if_exists(model, 'fc_norm')
    if hasattr(model, 'model') and isinstance(model.model, nn.Module):
        inner = model.model
        _append_if_exists(inner, 'cls_head')
        _append_if_exists(inner, 'head')
        _append_if_exists(inner, 'classifier')
        _append_if_exists(inner, 'fc_norm')
    uniq, seen = [], set()
    for module in candidates:
        if id(module) not in seen:
            uniq.append(module)
            seen.add(id(module))
    return uniq


def configure_linear_probing_trainable_params(model: nn.Module, mode_name: str) -> List[str]:
    if mode_name == 'CE':
        # CE 没有预训练表征，LP 协议不具备可比意义，这里跳过
        return []
    for p in model.parameters():
        p.requires_grad = False
    selected = get_linear_probe_modules(model)
    for module in selected:
        for p in module.parameters():
            p.requires_grad = True
    return [name for name, p in model.named_parameters() if p.requires_grad]


def build_finetune_optimizer(model: nn.Module, cfg: Dict):
    ft_lr = float(cfg.get('ft_lr', 1e-4))
    ft_wd = float(cfg.get('ft_weight_decay', 1e-5))
    backbone_lr_mult = float(cfg.get('backbone_lr_mult', 0.1))
    head_lr_mult = float(cfg.get('head_lr_mult', 1.0))
    if has_backbone_and_head(model):
        return optim.AdamW([
            {'params': model.backbone.parameters(), 'lr': ft_lr * backbone_lr_mult, 'weight_decay': ft_wd},
            {'params': model.cls_head.parameters(), 'lr': ft_lr * head_lr_mult, 'weight_decay': ft_wd},
        ])
    return optim.AdamW(model.parameters(), lr=ft_lr, weight_decay=ft_wd)


def build_protocol_optimizer(model: nn.Module, cfg: Dict, protocol: str, mode_name: str):
    if protocol == 'Linear_Probing':
        return optim.AdamW([p for p in model.parameters() if p.requires_grad], lr=float(cfg.get('lp_lr', 1e-3)), weight_decay=float(cfg.get('lp_weight_decay', 1e-4)))
    cfg_local = copy.deepcopy(cfg)
    if parse_few_shot_ratio(protocol) is not None:
        cfg_local['ft_lr'] = float(cfg.get('few_shot_ft_lr', cfg.get('ft_lr', 1e-4)))
    return build_finetune_optimizer(model, cfg_local)


def build_scheduler(optimizer, cfg: Dict, total_epochs: int):
    scheduler_name = cfg.get('scheduler', 'cosine')
    if scheduler_name == 'cosine':
        return optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=max(total_epochs, 1), eta_min=float(cfg.get('min_lr', 1e-6)))
    if scheduler_name in (None, 'none'):
        return None
    raise ValueError(f'Unsupported scheduler: {scheduler_name}')


def reinit_head(model: nn.Module):
    modules = get_linear_probe_modules(model)
    for m in modules:
        if hasattr(m, 'reset_parameters'):
            m.reset_parameters()


def train_with_protocol_and_val_selection(model: nn.Module, train_loader, val_loader, cfg: Dict, device: torch.device, class_names: List[str], csv_path: str, protocol: str, mode_name: str):
    if mode_name == 'CE' and protocol == 'Linear_Probing':
        print('   [Skip] CE 不参与 Linear_Probing（无预训练表征可冻结）。')
        return None, None, None, None

    if protocol == 'Linear_Probing':
        trainable_names = configure_linear_probing_trainable_params(model, mode_name)
        print(f'   [Linear_Probing] trainable_tensors={len(trainable_names)}')
        epochs = int(cfg.get('lp_epochs', 25))
    else:
        for p in model.parameters():
            p.requires_grad = True
        epochs = int(cfg.get('finetune_epochs', 40))

    optimizer = build_protocol_optimizer(model, cfg, protocol, mode_name)
    scheduler = build_scheduler(optimizer, cfg, epochs)
    pos_weight = compute_pos_weight(train_loader, device) if bool(cfg.get('use_pos_weight', True)) else None
    criterion = nn.BCEWithLogitsLoss(pos_weight=pos_weight)
    early_stop_patience = int(cfg.get('early_stop_patience', 8))
    grad_clip = float(cfg.get('grad_clip', 0.0))

    best_state = None
    best_epoch = -1
    best_val_score = -1e18
    best_thresholds = np.full(len(class_names), 0.5, dtype=np.float32)
    bad_epochs = 0

    for epoch in range(epochs):
        model.train()
        train_losses = []
        for batch in train_loader:
            x = batch_get_model_input(batch)
            y = batch_get_labels(batch)
            if isinstance(x, dict):
                x = {k: v.to(device, non_blocking=True) for k, v in x.items()}
            else:
                x = x.to(device, non_blocking=True)
            y = y.to(device, non_blocking=True)
            optimizer.zero_grad()
            logits = model_forward_cls(model, x)
            loss = criterion(logits, y)
            loss.backward()
            if grad_clip > 0:
                torch.nn.utils.clip_grad_norm_(model.parameters(), grad_clip)
            optimizer.step()
            train_losses.append(float(loss.item()))

        val_probs, val_targets = collect_probs(model, val_loader, device)
        thresholds = tune_thresholds_per_class(val_probs, val_targets) if bool(cfg.get('tune_thresholds', True)) else np.full(len(class_names), 0.5, dtype=np.float32)
        val_metrics = evaluate_from_probs(val_probs, val_targets, class_names, thresholds, csv_path)
        score = float(val_metrics[cfg.get('monitor_metric', 'AUPRC')])
        print(
            f"Epoch {epoch+1:02d} | train_loss={np.mean(train_losses):.4f} | "
            f"val_AUPRC={val_metrics['AUPRC']:.4f} | val_Macro_AUC={val_metrics['Macro_AUC']:.4f} | "
            f"val_Macro_F1={val_metrics['Macro_F1']:.4f} | val_MI_F1={val_metrics['MI_F1']:.4f}"
        )

        if score > best_val_score:
            best_val_score = score
            best_epoch = epoch + 1
            best_state = {k: v.detach().cpu() for k, v in model.state_dict().items()}
            best_thresholds = thresholds.copy()
            bad_epochs = 0
        else:
            bad_epochs += 1

        if scheduler is not None:
            scheduler.step()
        if bad_epochs >= early_stop_patience:
            print(f'   [Early Stop] patience={early_stop_patience}, best_epoch={best_epoch}, best_val_score={best_val_score:.4f}')
            break

    if best_state is None:
        raise RuntimeError('训练过程中未保存到有效 best_state。')
    model.load_state_dict(best_state)
    return model, best_thresholds, best_epoch, best_val_score


# ==============================================================================
# 7. 单个 seed 跑所有协议
# ==============================================================================
def run_seed_all_protocols(mode_name: str, seed: int, cfg: Dict, data_dir: str, csv_path: str, protocols: List[str]):
    print(f"\n🚀 [Seed {seed}] {mode_name}")
    seed_everything(seed)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    # 预训练只用 full-train split 一次
    pretrain_loader, _, _, _, _ = prepare_protocol_loaders(data_dir, cfg, 'Full_Finetune', seed)
    pretrain_model, backbone_dim = build_pretrain_model(mode_name, device, cfg, num_classes=len(CLASS_NAMES))
    pretrained_state = pretrain_and_get_state(mode_name, pretrain_model, backbone_dim, pretrain_loader, cfg, device)
    del pretrain_model
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    rows = []
    for protocol in protocols:
        _, train_loader, val_loader, test_loader, few_shot_indices = prepare_protocol_loaders(data_dir, cfg, protocol, seed)
        model, _ = build_downstream_model(mode_name, device, cfg, num_classes=len(CLASS_NAMES))
        if pretrained_state is not None:
            model.load_state_dict(pretrained_state, strict=False)
        reinit_head(model)
        result = train_with_protocol_and_val_selection(model, train_loader, val_loader, cfg, device, CLASS_NAMES, csv_path, protocol, mode_name)
        if result[0] is None:
            continue
        model, best_thresholds, best_epoch, best_val_score = result

        val_probs, val_targets = collect_probs(model, val_loader, device)
        test_probs, test_targets = collect_probs(model, test_loader, device)
        val_metrics = evaluate_from_probs(val_probs, val_targets, CLASS_NAMES, best_thresholds, csv_path)
        test_metrics = evaluate_from_probs(test_probs, test_targets, CLASS_NAMES, best_thresholds, csv_path)
        rows.append({
            'mode_name': mode_name,
            'protocol': protocol,
            'seed': seed,
            'n_train_samples': len(train_loader.dataset),
            'best_epoch': best_epoch,
            'best_val_score': best_val_score,
            **{f'val_{k}': v for k, v in val_metrics.items()},
            **{f'test_{k}': v for k, v in test_metrics.items()},
            'thresholds': json.dumps(best_thresholds.tolist(), ensure_ascii=False),
            'few_shot_indices': json.dumps(few_shot_indices, ensure_ascii=False) if few_shot_indices is not None else '',
            'config': json.dumps(cfg, ensure_ascii=False, sort_keys=True),
        })

    return rows


# ==============================================================================
# 8. 汇总与主程序
# ==============================================================================
def summarize_protocol(df: pd.DataFrame, save_dir: str, protocol: str):
    sub = df[df['protocol'] == protocol]
    if sub.empty:
        return
    metric_cols = ['test_Macro_AUC', 'test_AUPRC', 'test_Macro_F1', 'test_MI_F1', 'test_HNDR_Pair', 'test_HNDR_Inst']
    rows = []
    for mode_name, g in sub.groupby('mode_name'):
        mean_series = g[metric_cols].mean()
        std_series = g[metric_cols].std(ddof=0)
        rows.append({
            'mode_name': mode_name,
            'Macro_AUC': f"{mean_series['test_Macro_AUC']:.4f} ± {std_series['test_Macro_AUC']:.4f}",
            'AUPRC': f"{mean_series['test_AUPRC']:.4f} ± {std_series['test_AUPRC']:.4f}",
            'F1': f"{mean_series['test_Macro_F1']:.4f} ± {std_series['test_Macro_F1']:.4f}",
            'MI_F1': f"{mean_series['test_MI_F1']:.4f} ± {std_series['test_MI_F1']:.4f}",
            'HNDR_Pair': f"{mean_series['test_HNDR_Pair']:.4f} ± {std_series['test_HNDR_Pair']:.4f}",
            'HNDR_Inst': f"{mean_series['test_HNDR_Inst']:.4f} ± {std_series['test_HNDR_Inst']:.4f}",
        })
    out_df = pd.DataFrame(rows).sort_values('mode_name')
    out_df.to_csv(os.path.join(save_dir, f'master_table_{protocol}.csv'), index=False)


def build_argparser():
    parser = argparse.ArgumentParser(description='Fair comparison script for CE/SimCLR/HCL/SupCon/MTAE/ST_MEM aligned to Stage8/10 shell')
    parser.add_argument('--data-dir', type=str, default=DEFAULT_DATA_DIR)
    parser.add_argument('--save-dir', type=str, default=DEFAULT_SAVE_DIR)
    parser.add_argument('--csv-path', type=str, default='')
    parser.add_argument('--seeds', type=int, nargs='*', default=DEFAULT_SEEDS)
    parser.add_argument('--protocols', type=str, nargs='*', default=DEFAULT_PROTOCOLS)
    parser.add_argument('--models', type=str, nargs='*', default=DEFAULT_MODELS)
    return parser


def main():
    args = build_argparser().parse_args()
    ensure_dir(args.save_dir)
    csv_path = resolve_confusable_csv_path(args.csv_path or None)

    print('\n' + '▼' * 80)
    print('🔬 启动公平评估：CE / SimCLR / HCL / SupCon / MTAE / ST_MEM')
    print('   壳口径尽量与 Stage8/Stage10 对齐：processed_v3 / 5 protocols / 42,123,1024 / AUPRC选模 / tuned thresholds / HNDR')
    print(f'   Models    = {args.models}')
    print(f'   Protocols = {args.protocols}')
    print(f'   Seeds     = {args.seeds}')
    print(f'   CSV       = {csv_path}')
    print('▲' * 80)

    all_rows = []
    for mode_name in args.models:
        if mode_name not in MODEL_CONFIGS:
            raise ValueError(f'Unsupported mode: {mode_name}')
        cfg = copy.deepcopy(MODEL_CONFIGS[mode_name])
        print(f"\n{'=' * 30} Evaluating {mode_name} {'=' * 30}")
        for seed in args.seeds:
            rows = run_seed_all_protocols(mode_name, seed, cfg, args.data_dir, csv_path, args.protocols)
            all_rows.extend(rows)
            pd.DataFrame(all_rows).to_csv(os.path.join(args.save_dir, 'formal_eval_per_seed_results_all_protocols.csv'), index=False)

    df = pd.DataFrame(all_rows)
    for protocol in args.protocols:
        summarize_protocol(df, args.save_dir, protocol)

    summary = {}
    for protocol in args.protocols:
        sub = df[df['protocol'] == protocol]
        if sub.empty:
            continue
        summary[protocol] = {}
        for mode_name, g in sub.groupby('mode_name'):
            metric_cols = ['test_Macro_AUC', 'test_AUPRC', 'test_Macro_F1', 'test_MI_F1', 'test_HNDR_Pair', 'test_HNDR_Inst']
            mean_series = g[metric_cols].mean()
            std_series = g[metric_cols].std(ddof=0)
            summary[protocol][mode_name] = {col.replace('test_', ''): f'{mean_series[col]:.4f} ± {std_series[col]:.4f}' for col in metric_cols}
    with open(os.path.join(args.save_dir, 'summary.json'), 'w', encoding='utf-8') as f:
        json.dump(summary, f, ensure_ascii=False, indent=2)

    print(f"✅ 已保存到: {args.save_dir}")


if __name__ == '__main__':
    main()
