#!/usr/bin/env python3
# -*- coding: utf-8 -*-

from pathlib import Path
import argparse
import copy
import json
import os
import random
import sys
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset, Subset
from sklearn.metrics import accuracy_score, average_precision_score, f1_score, roc_auc_score


PROJECT_ROOT = Path("/root/MCKI_Project")
SRC4_ROOT = PROJECT_ROOT / "src4"
SRC3_ROOT = PROJECT_ROOT / "src3"

for _p in [PROJECT_ROOT, SRC4_ROOT, SRC3_ROOT]:
    if _p.exists() and str(_p) not in sys.path:
        sys.path.insert(0, str(_p))

from MCKI_backbone_factory import build_MCKI_backbone  # noqa: E402


CLASS_NAMES = ["NORM", "MI", "STTC", "CD", "HYP"]
MI_INDEX = CLASS_NAMES.index("MI")

DATA_DIR = PROJECT_ROOT / "data" / "processed_v3"
EXTERNAL_DIR = PROJECT_ROOT / "data" / "external_processed" / "sph_5class"
RESULT_ROOT = PROJECT_ROOT / "results" / "external_eval" / "sph" / "MCKI_stage8_true"
RESULT_ROOT.mkdir(parents=True, exist_ok=True)

DEFAULT_HARD_PAIRS = PROJECT_ROOT / "src3" / "confusable_pairs_v1.csv"
DEFAULT_STAGE8_EXPORT_DIR = PROJECT_ROOT / "formal_eval_outputs_stage8_pairaware_leadaware_multiscale_MCKI"

STAGE8_EVAL_CFG = {
    "backbone_name": "resnet18",
    "batch_size": 64,
    "num_workers": 4,
    "pin_memory": True,
    "ft_lr": 1e-4,
    "ft_weight_decay": 1e-5,
    "backbone_lr_mult": 0.1,
    "head_lr_mult": 1.0,
    "lp_lr": 1e-3,
    "lp_weight_decay": 1e-4,
    "lp_epochs": 25,
    "finetune_epochs": 40,
    "monitor_metric": "AUPRC",
    "early_stop_patience": 8,
    "scheduler": "cosine",
    "tune_thresholds": True,
    "use_pos_weight": True,
    # ===== true Stage8 lead-aware path =====
    "use_lead_aware_input": True,
    "lead_emb_dim": 16,
    "lead_adapter_dropout": 0.0,
}

FEWSHOT_RATIOS = {
    "Few_Shot_1%": 0.01,
    "Few_Shot_10%": 0.10,
    "Few_Shot_25%": 0.25,
}
ALL_PROTOCOLS = ["Linear_Probing", "Few_Shot_1%", "Few_Shot_10%", "Few_Shot_25%", "Full_Finetune"]
SUMMARY_KEYS = ["Macro_AUC", "AUPRC", "Macro_F1", "MI_F1", "HNDR_Pair", "HNDR_Inst"]


def normalize_protocol_name(protocol: str) -> str:
    alias = {
        "ALL": "ALL",
        "all": "ALL",
        "Full_Finetuning": "Full_Finetune",
        "Full_FineTune": "Full_Finetune",
        "Full_Fineline": "Full_Finetune",
        "full_fineline": "Full_Finetune",
        "full_finetune": "Full_Finetune",
        "linear_probing": "Linear_Probing",
        "few_shot_1%": "Few_Shot_1%",
        "few_shot_10%": "Few_Shot_10%",
        "few_shot_25%": "Few_Shot_25%",
    }
    return alias.get(protocol, protocol)


def seed_everything(seed: int):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


class InternalNPYDataset(Dataset):
    def __init__(self, x_path: Path, y_path: Path):
        self.x = np.load(x_path, mmap_mode="r")
        self.y = np.load(y_path, mmap_mode="r")

    def __len__(self):
        return len(self.y)

    def __getitem__(self, idx: int):
        x = torch.tensor(self.x[idx], dtype=torch.float32).permute(1, 0)  # (12, 1000)
        y = torch.tensor(self.y[idx], dtype=torch.float32)
        return x, y


class ExternalNPYDataset(Dataset):
    def __init__(self, x_path: Path, y_path: Path):
        self.x = np.load(x_path, mmap_mode="r")
        self.y = np.load(y_path, mmap_mode="r")

    def __len__(self):
        return len(self.y)

    def __getitem__(self, idx: int):
        x = torch.tensor(self.x[idx], dtype=torch.float32).permute(1, 0)  # (12, 1000)
        y = torch.tensor(self.y[idx], dtype=torch.float32)
        return x, y


class LeadAwareInputAdapter(nn.Module):
    """Stage8 lead-aware adapter."""

    def __init__(self, num_leads: int = 12, emb_dim: int = 16, dropout: float = 0.0):
        super().__init__()
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
        lead_ids = torch.arange(c, device=x.device)
        emb = self.embedding(lead_ids)
        scale = self.to_scale(emb).view(1, c, 1)
        bias = self.to_bias(emb).view(1, c, 1)
        out = x * (1.0 + scale) + bias
        return self.dropout(out)


def attach_lead_adapter_if_needed(model: nn.Module, cfg: Dict):
    if cfg.get("use_lead_aware_input", False):
        model.lead_adapter = LeadAwareInputAdapter(
            num_leads=12,
            emb_dim=int(cfg.get("lead_emb_dim", 16)),
            dropout=float(cfg.get("lead_adapter_dropout", 0.0)),
        )
    else:
        model.lead_adapter = None
    return model


def wrap_forward_cls_for_leadaware(model: nn.Module):
    if not hasattr(model, "_original_forward_cls"):
        model._original_forward_cls = model.forward_cls

    def _forward_cls_leadaware(x: torch.Tensor):
        if hasattr(model, "lead_adapter") and model.lead_adapter is not None:
            x = model.lead_adapter(x)
        return model._original_forward_cls(x)

    model.forward_cls = _forward_cls_leadaware
    return model


def build_model(cfg: Dict, device: torch.device) -> nn.Module:
    model, _ = build_MCKI_backbone(
        cfg["backbone_name"],
        num_classes=len(CLASS_NAMES),
        device=device,
        cfg=cfg,
    )
    model = attach_lead_adapter_if_needed(model, cfg)
    model = wrap_forward_cls_for_leadaware(model)
    return model.to(device)


def reinit_head(model: nn.Module):
    if hasattr(model, "cls_head"):
        if isinstance(model.cls_head, nn.Linear):
            model.cls_head.reset_parameters()
        else:
            for module in model.cls_head.modules():
                if hasattr(module, "reset_parameters"):
                    module.reset_parameters()


def set_backbone_trainable(model: nn.Module, trainable: bool):
    for p in model.backbone.parameters():
        p.requires_grad = trainable
    if hasattr(model, "lead_adapter") and model.lead_adapter is not None:
        for p in model.lead_adapter.parameters():
            p.requires_grad = trainable


def resolve_pretrained_for_seed(seed: int, override: Optional[str] = None) -> Path:
    if override:
        return Path(override)

    candidates = [
        DEFAULT_STAGE8_EXPORT_DIR / f"seed_{seed}" / f"stage8_pretrained_backbone_seed{seed}.pth",
        DEFAULT_STAGE8_EXPORT_DIR / f"seed_{seed}" / "pretrained_backbone.pth",
        PROJECT_ROOT / "checkpoints" / f"stage8_pretrained_backbone_seed{seed}.pth",
        PROJECT_ROOT / "checkpoints" / f"stage8_leadaware_multiscale_seed{seed}.pth",
    ]
    for path in candidates:
        if path.exists():
            return path
    return candidates[0]


def extract_state_dict(raw_obj):
    if isinstance(raw_obj, dict):
        for key in ["model_state_dict", "state_dict", "model", "net", "weights"]:
            if key in raw_obj and isinstance(raw_obj[key], dict):
                return raw_obj[key]
    if isinstance(raw_obj, dict):
        return raw_obj
    raise TypeError(f"Unsupported checkpoint format: {type(raw_obj)}")


def sanitize_state_dict_keys(state_dict: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
    cleaned = {}
    for k, v in state_dict.items():
        nk = k
        if nk.startswith("module."):
            nk = nk[len("module."):]
        if nk.startswith("model."):
            nk = nk[len("model."):]
        cleaned[nk] = v
    return cleaned


def load_pretrained_robust(model: nn.Module, ckpt_path: Path, device: torch.device):
    if not ckpt_path.exists():
        raise FileNotFoundError(f"Checkpoint not found: {ckpt_path}")

    raw = torch.load(str(ckpt_path), map_location=device, weights_only=False)
    state_dict = sanitize_state_dict_keys(extract_state_dict(raw))
    load_res = model.load_state_dict(state_dict, strict=False)

    missing = list(load_res.missing_keys)
    unexpected = list(load_res.unexpected_keys)
    print(f"[INFO] Loaded checkpoint: {ckpt_path}")
    if missing:
        print(f"[INFO] Missing keys ({len(missing)}): {missing[:20]}")
    if unexpected:
        print(f"[INFO] Unexpected keys ({len(unexpected)}): {unexpected[:20]}")
    return model


def build_loader(dataset, batch_size: int, shuffle: bool, num_workers: int, seed: Optional[int] = None):
    generator = None
    if shuffle:
        generator = torch.Generator()
        generator.manual_seed(int(seed if seed is not None else 0))
    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=num_workers,
        pin_memory=torch.cuda.is_available(),
        drop_last=False,
        generator=generator,
    )


def compute_pos_weight_from_loader(loader, device: torch.device) -> torch.Tensor:
    ys = []
    for _, y in loader:
        ys.append(y.float())
    y = torch.cat(ys, dim=0)
    pos = y.sum(dim=0)
    neg = y.shape[0] - pos
    pos_weight = (neg / (pos + 1e-6)).clamp(min=1.0, max=20.0)
    return pos_weight.to(device)


def build_scheduler(optimizer, total_epochs: int):
    return optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=max(total_epochs, 1), eta_min=1e-6)


def build_internal_loaders(protocol: str, seed: int, batch_size: int, num_workers: int):
    protocol = normalize_protocol_name(protocol)
    train_ds = InternalNPYDataset(DATA_DIR / "X_train.npy", DATA_DIR / "y_train_mh.npy")
    val_ds = InternalNPYDataset(DATA_DIR / "X_val.npy", DATA_DIR / "y_val_mh.npy")

    full_n = len(train_ds)
    if protocol in FEWSHOT_RATIOS:
        ratio = FEWSHOT_RATIOS[protocol]
        rng = np.random.default_rng(seed + 2026)
        k = max(1, int(round(full_n * ratio)))
        idx = sorted(rng.choice(full_n, size=k, replace=False).tolist())
        train_ds = Subset(train_ds, idx)
        print(f"[INFO] {protocol}: using {k}/{full_n} training samples (ratio={ratio:.2%})")
    elif protocol == "Full_Finetune":
        print(f"[INFO] Full_Finetune: using full training set {full_n}/{full_n}")

    train_loader = build_loader(train_ds, batch_size=batch_size, shuffle=True, num_workers=num_workers, seed=seed + 23)
    val_loader = build_loader(val_ds, batch_size=batch_size, shuffle=False, num_workers=num_workers)
    return train_loader, val_loader


def build_external_loader(batch_size: int, num_workers: int):
    ds = ExternalNPYDataset(EXTERNAL_DIR / "X_test.npy", EXTERNAL_DIR / "y_test_mh.npy")
    return DataLoader(
        ds,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=torch.cuda.is_available(),
    )


@torch.no_grad()
def eval_loader(model: nn.Module, loader, device: torch.device) -> Tuple[np.ndarray, np.ndarray]:
    model.eval()
    all_probs, all_targets = [], []
    for x, y in loader:
        x = x.to(device, non_blocking=True)
        y = y.to(device, non_blocking=True)
        logits = model.forward_cls(x)
        probs = torch.sigmoid(logits).cpu().numpy()
        all_probs.append(probs)
        all_targets.append(y.cpu().numpy())
    return np.vstack(all_probs), np.vstack(all_targets)


def find_best_thresholds(val_probs: np.ndarray, val_targets: np.ndarray, num_classes: int, grid=None):
    if grid is None:
        grid = np.arange(0.10, 0.91, 0.05)
    thresholds = np.zeros(num_classes, dtype=np.float32)
    per_class_val_f1 = np.zeros(num_classes, dtype=np.float32)

    for c in range(num_classes):
        y_true = val_targets[:, c]
        y_prob = val_probs[:, c]
        best_t = 0.5
        best_f1 = -1.0
        for t in grid:
            y_pred = (y_prob >= t).astype(int)
            score = f1_score(y_true, y_pred, zero_division=0)
            if score > best_f1:
                best_f1 = score
                best_t = t
        thresholds[c] = best_t
        per_class_val_f1[c] = best_f1
    return thresholds, per_class_val_f1


def calculate_hndr(probs: np.ndarray, targets: np.ndarray, class_names: List[str], csv_path: str):
    if not os.path.exists(csv_path):
        return None, None, {}

    pairs_df = pd.read_csv(csv_path)
    pair_metrics = {}
    total_hard_samples = 0
    correct_hard_samples = 0

    for _, row in pairs_df.iterrows():
        disease_a = str(row["disease_a"]).strip()
        disease_b = str(row["disease_b"]).strip()
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

        pair_metrics[f"{disease_a}_vs_{disease_b}"] = acc
        total_hard_samples += len(binary_targets)
        correct_hard_samples += int(np.sum(binary_preds == binary_targets))

    if not pair_metrics:
        return None, None, {}

    hndr_pair = float(np.mean(list(pair_metrics.values())))
    hndr_inst = float(correct_hard_samples / total_hard_samples) if total_hard_samples > 0 else 0.0
    return hndr_pair, hndr_inst, pair_metrics


def compute_metrics_raw(probs: np.ndarray, targets: np.ndarray) -> Dict[str, float]:
    return {
        "Macro_AUC": float(roc_auc_score(targets, probs, average="macro")),
        "AUPRC": float(average_precision_score(targets, probs, average="macro")),
        "Macro_F1@0.5": float(f1_score(targets, (probs >= 0.5).astype(int), average="macro", zero_division=0)),
        "MI_F1@0.5": float(f1_score(targets[:, MI_INDEX], (probs[:, MI_INDEX] >= 0.5).astype(int), zero_division=0)),
    }


def compute_metrics_with_thresholds(probs: np.ndarray, targets: np.ndarray, thresholds: np.ndarray) -> Dict[str, float]:
    preds = (probs >= thresholds.reshape(1, -1)).astype(int)
    return {
        "Macro_F1": float(f1_score(targets, preds, average="macro", zero_division=0)),
        "MI_F1": float(f1_score(targets[:, MI_INDEX], preds[:, MI_INDEX], zero_division=0)),
    }


def build_optimizer_for_protocol(model: nn.Module, protocol: str, cfg: Dict):
    protocol = normalize_protocol_name(protocol)
    if protocol == "Linear_Probing":
        if hasattr(model, "lead_adapter") and model.lead_adapter is not None:
            for p in model.lead_adapter.parameters():
                p.requires_grad = False
        return optim.AdamW(
            model.cls_head.parameters(),
            lr=float(cfg["lp_lr"]),
            weight_decay=float(cfg["lp_weight_decay"]),
        )

    ft_lr = float(cfg["ft_lr"])
    ft_wd = float(cfg["ft_weight_decay"])
    backbone_lr_mult = float(cfg["backbone_lr_mult"])
    head_lr_mult = float(cfg["head_lr_mult"])

    param_groups = [
        {
            "params": model.backbone.parameters(),
            "lr": ft_lr * backbone_lr_mult,
            "weight_decay": ft_wd,
        },
        {
            "params": model.cls_head.parameters(),
            "lr": ft_lr * head_lr_mult,
            "weight_decay": ft_wd,
        },
    ]
    if hasattr(model, "lead_adapter") and model.lead_adapter is not None:
        param_groups.append(
            {
                "params": model.lead_adapter.parameters(),
                "lr": ft_lr * head_lr_mult,
                "weight_decay": ft_wd,
            }
        )
    return optim.AdamW(param_groups)


def prepare_model_for_protocol(protocol: str, pretrained_ckpt: Path, cfg: Dict, device: torch.device) -> nn.Module:
    protocol = normalize_protocol_name(protocol)
    model = build_model(cfg, device)
    model = load_pretrained_robust(model, pretrained_ckpt, device)

    # Fair transfer: always reset the classifier head.
    reinit_head(model)

    if protocol == "Linear_Probing":
        set_backbone_trainable(model, False)
        for p in model.cls_head.parameters():
            p.requires_grad = True
        print("[INFO] Protocol = Linear_Probing (backbone + lead_adapter frozen, cls_head reset)")
    elif protocol in FEWSHOT_RATIOS or protocol == "Full_Finetune":
        set_backbone_trainable(model, True)
        for p in model.cls_head.parameters():
            p.requires_grad = True
        print(f"[INFO] Protocol = {protocol} (all trainable, cls_head reset)")
    else:
        raise ValueError(f"Unsupported protocol: {protocol}")

    return model


def train_one_epoch(model: nn.Module, loader, optimizer, criterion, device: torch.device) -> float:
    model.train()
    total_loss = 0.0
    total_n = 0

    for x, y in loader:
        x = x.to(device, non_blocking=True)
        y = y.to(device, non_blocking=True)

        optimizer.zero_grad()
        logits = model.forward_cls(x)
        loss = criterion(logits, y)
        loss.backward()
        optimizer.step()

        bs = y.shape[0]
        total_loss += float(loss.item()) * bs
        total_n += bs

    return total_loss / max(total_n, 1)


def get_artifact_path(protocol: str, seed: int) -> Path:
    artifact_dir = RESULT_ROOT / "artifacts"
    artifact_dir.mkdir(parents=True, exist_ok=True)
    return artifact_dir / f"{protocol}_seed{seed}.pth"


def fit_protocol(
    protocol: str,
    seed: int,
    pretrained_ckpt: Path,
    batch_size: int,
    num_workers: int,
    device: torch.device,
    hard_pairs_csv: str,
    force_retrain: bool = False,
):
    protocol = normalize_protocol_name(protocol)
    artifact_path = get_artifact_path(protocol, seed)
    cfg = copy.deepcopy(STAGE8_EVAL_CFG)

    if artifact_path.exists() and (not force_retrain):
        print(f"[INFO] Reusing trained protocol artifact: {artifact_path}")
        model = build_model(cfg, device)
        obj = torch.load(artifact_path, map_location=device, weights_only=False)
        model.load_state_dict(obj["model_state_dict"], strict=True)
        thresholds = np.asarray(obj["thresholds"], dtype=np.float32)
        return model, thresholds, artifact_path

    train_loader, val_loader = build_internal_loaders(protocol, seed, batch_size, num_workers)
    model = prepare_model_for_protocol(protocol, pretrained_ckpt, cfg, device)

    max_epochs = int(cfg["lp_epochs"] if protocol == "Linear_Probing" else cfg["finetune_epochs"])
    patience_limit = int(cfg["early_stop_patience"])
    optimizer = build_optimizer_for_protocol(model, protocol, cfg)
    pos_weight = compute_pos_weight_from_loader(train_loader, device) if cfg.get("use_pos_weight", True) else None
    criterion = nn.BCEWithLogitsLoss(pos_weight=pos_weight)
    scheduler = build_scheduler(optimizer, max_epochs)

    best_val_score = -1.0
    best_state = None
    best_thresholds = np.full(len(CLASS_NAMES), 0.5, dtype=np.float32)
    best_per_class_val_f1 = np.zeros(len(CLASS_NAMES), dtype=np.float32)
    patience = 0

    print(f"[INFO] Start fitting protocol={protocol} | seed={seed}")
    for epoch in range(1, max_epochs + 1):
        train_loss = train_one_epoch(model, train_loader, optimizer, criterion, device)
        scheduler.step()

        val_probs, val_targets = eval_loader(model, val_loader, device)
        val_thresholds, per_class_val_f1 = find_best_thresholds(val_probs, val_targets, len(CLASS_NAMES))
        val_metrics_raw = compute_metrics_raw(val_probs, val_targets)
        val_metrics_th = compute_metrics_with_thresholds(val_probs, val_targets, val_thresholds)
        val_score = float(val_metrics_raw[cfg["monitor_metric"]])

        print(
            f"[Epoch {epoch:02d}/{max_epochs}] "
            f"train_loss={train_loss:.6f}  "
            f"val_AUPRC={val_metrics_raw['AUPRC']:.6f}  "
            f"val_Macro_AUC={val_metrics_raw['Macro_AUC']:.6f}  "
            f"val_Macro_F1={val_metrics_th['Macro_F1']:.6f}  "
            f"val_MI_F1={val_metrics_th['MI_F1']:.6f}"
        )

        if val_score > best_val_score:
            best_val_score = val_score
            best_state = copy.deepcopy(model.state_dict())
            best_thresholds = val_thresholds.astype(np.float32)
            best_per_class_val_f1 = per_class_val_f1.astype(np.float32)
            patience = 0
        else:
            patience += 1

        if patience >= patience_limit:
            print(f"[INFO] Early stopping triggered (patience={patience_limit})")
            break

    if best_state is None:
        raise RuntimeError(f"No best state captured for protocol={protocol}, seed={seed}")

    model.load_state_dict(best_state, strict=True)

    obj = {
        "protocol": protocol,
        "seed": seed,
        "thresholds": best_thresholds,
        "per_class_val_f1": best_per_class_val_f1,
        "best_val_score": float(best_val_score),
        "model_state_dict": model.state_dict(),
        "class_names": CLASS_NAMES,
        "stage8_eval_cfg": cfg,
        "pretrained_ckpt": str(pretrained_ckpt),
        "hard_pairs_csv": str(hard_pairs_csv),
    }
    torch.save(obj, artifact_path)
    print(f"[OK] Saved protocol artifact -> {artifact_path}")
    return model, best_thresholds, artifact_path


def save_json(obj, path: Path):
    with open(path, "w", encoding="utf-8") as f:
        json.dump(obj, f, ensure_ascii=False, indent=2)


def run_single(
    protocol: str,
    seed: int,
    pretrained_ckpt: Optional[str],
    batch_size: int,
    num_workers: int,
    device: torch.device,
    hard_pairs_csv: str,
    force_retrain: bool = False,
):
    resolved_ckpt = resolve_pretrained_for_seed(seed, pretrained_ckpt)
    model, thresholds, artifact_path = fit_protocol(
        protocol=protocol,
        seed=seed,
        pretrained_ckpt=resolved_ckpt,
        batch_size=batch_size,
        num_workers=num_workers,
        device=device,
        hard_pairs_csv=hard_pairs_csv,
        force_retrain=force_retrain,
    )

    ext_loader = build_external_loader(batch_size, num_workers)
    probs, targets = eval_loader(model, ext_loader, device)

    raw_metrics = compute_metrics_raw(probs, targets)
    th_metrics = compute_metrics_with_thresholds(probs, targets, thresholds)
    hndr_pair, hndr_inst, pair_metrics = calculate_hndr(probs, targets, CLASS_NAMES, hard_pairs_csv)

    final_metrics = {
        "Protocol": protocol,
        "Seed": int(seed),
        "Macro_AUC": float(raw_metrics["Macro_AUC"]),
        "AUPRC": float(raw_metrics["AUPRC"]),
        "Macro_F1": float(th_metrics["Macro_F1"]),
        "MI_F1": float(th_metrics["MI_F1"]),
        "Macro_F1@0.5": float(raw_metrics["Macro_F1@0.5"]),
        "MI_F1@0.5": float(raw_metrics["MI_F1@0.5"]),
        "HNDR_Pair": None if hndr_pair is None else float(hndr_pair),
        "HNDR_Inst": None if hndr_inst is None else float(hndr_inst),
        "Artifact_Path": str(artifact_path),
        "Hard_Pairs_CSV": str(hard_pairs_csv),
        "Pretrained_CKPT": str(resolved_ckpt),
    }

    run_dir = RESULT_ROOT / f"{protocol}_seed{seed}"
    run_dir.mkdir(parents=True, exist_ok=True)
    np.save(run_dir / "external_probs.npy", probs)
    np.save(run_dir / "external_targets.npy", targets)
    np.save(run_dir / "thresholds.npy", thresholds.astype(np.float32))
    save_json(final_metrics, run_dir / "metrics.json")
    save_json({k: float(v) for k, v in pair_metrics.items()}, run_dir / "pair_metrics.json")

    print("\n=== SPH External MCKI Stage8 True Results ===")
    for k, v in final_metrics.items():
        if isinstance(v, float):
            print(f"{k}: {v:.6f}")
        else:
            print(f"{k}: {v}")
    print(f"[OK] Saved results -> {run_dir}\n")

    return final_metrics, run_dir


def summarize_runs(all_metrics: List[Dict], summary_dir: Path):
    if not all_metrics:
        return

    summary_dir.mkdir(parents=True, exist_ok=True)
    df = pd.DataFrame(all_metrics)
    df.to_csv(summary_dir / "per_run_metrics.csv", index=False)

    rows = []
    for protocol in ALL_PROTOCOLS:
        sub = df[df["Protocol"] == protocol]
        if len(sub) == 0:
            continue
        row = {"Protocol": protocol, "Num_Runs": int(len(sub))}
        for metric in SUMMARY_KEYS:
            vals = pd.to_numeric(sub[metric], errors="coerce").dropna()
            if len(vals) == 0:
                row[metric] = "NA"
            elif len(vals) == 1:
                row[metric] = f"{vals.mean():.4f} ± 0.0000"
            else:
                row[metric] = f"{vals.mean():.4f} ± {vals.std(ddof=1):.4f}"
        rows.append(row)

    summary_df = pd.DataFrame(rows)
    summary_df.to_csv(summary_dir / "summary_by_protocol.csv", index=False)
    save_json(rows, summary_dir / "summary_by_protocol.json")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--protocol",
        type=str,
        default="Linear_Probing",
        choices=ALL_PROTOCOLS + ["ALL", "Full_Finetuning", "Full_Fineline", "full_fineline"],
    )
    parser.add_argument("--seeds", type=int, nargs="+", default=[42])
    parser.add_argument("--batch-size", type=int, default=STAGE8_EVAL_CFG["batch_size"])
    parser.add_argument("--num-workers", type=int, default=STAGE8_EVAL_CFG["num_workers"])
    parser.add_argument(
        "--pretrained-ckpt",
        type=str,
        default="",
        help="Optional single checkpoint override. If empty, resolve per-seed exported Stage8 backbone automatically.",
    )
    parser.add_argument("--hard-pairs-csv", type=str, default=str(DEFAULT_HARD_PAIRS))
    parser.add_argument("--force-retrain", action="store_true")
    args = parser.parse_args()

    protocol = normalize_protocol_name(args.protocol)
    protocols = ALL_PROTOCOLS if protocol == "ALL" else [protocol]
    seeds = [int(s) for s in args.seeds]

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("[INFO] device =", device)
    print("[INFO] protocols =", protocols)
    print("[INFO] seeds =", seeds)
    print("[INFO] hard_pairs_csv =", args.hard_pairs_csv)
    if args.pretrained_ckpt:
        print("[INFO] pretrained_ckpt override =", args.pretrained_ckpt)
    else:
        print("[INFO] pretrained_ckpt override = <auto per seed from exported Stage8 backbones>")

    all_metrics = []
    summary_dir = RESULT_ROOT / "summary"

    for seed in seeds:
        seed_everything(seed)
        resolved_ckpt = resolve_pretrained_for_seed(seed, args.pretrained_ckpt or None)
        print(f"[INFO] seed={seed} resolved_pretrained_ckpt={resolved_ckpt}")
        for protocol_name in protocols:
            final_metrics, _ = run_single(
                protocol=protocol_name,
                seed=seed,
                pretrained_ckpt=args.pretrained_ckpt or None,
                batch_size=args.batch_size,
                num_workers=args.num_workers,
                device=device,
                hard_pairs_csv=args.hard_pairs_csv,
                force_retrain=args.force_retrain,
            )
            all_metrics.append(final_metrics)
            summarize_runs(all_metrics, summary_dir)

    print(f"[OK] Summary saved -> {summary_dir}")


if __name__ == "__main__":
    main()
