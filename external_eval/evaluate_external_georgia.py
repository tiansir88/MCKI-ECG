#!/usr/bin/env python3
# -*- coding: utf-8 -*-

from pathlib import Path
import os
import sys
import json
import copy
import argparse
import random
import csv
import math

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader, Subset
from sklearn.metrics import roc_auc_score, average_precision_score, f1_score, accuracy_score

PROJECT_ROOT = Path("/root/MCKI_Project")
SRC4_ROOT = PROJECT_ROOT / "src4"

if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))
if str(SRC4_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC4_ROOT))

from src4.train_v3_2stage import TwoStageModel


# =========================================================
# Robust checkpoint loader
# =========================================================
def load_checkpoint_robust(model, ckpt_path, device):
    ckpt_path = Path(ckpt_path)
    if not ckpt_path.exists():
        raise FileNotFoundError(f"Checkpoint not found: {ckpt_path}")

    obj = torch.load(str(ckpt_path), map_location=device, weights_only=False)

    state_dict = None
    if isinstance(obj, dict):
        for key in ["model_state_dict", "state_dict", "net", "model", "checkpoint"]:
            if key in obj and isinstance(obj[key], dict):
                state_dict = obj[key]
                break

        if state_dict is None and len(obj) > 0:
            if all(torch.is_tensor(v) for v in obj.values()):
                state_dict = obj

    if state_dict is None:
        raise RuntimeError(
            f"Cannot parse checkpoint state_dict from: {ckpt_path}. "
            f"Available top-level keys: {list(obj.keys()) if isinstance(obj, dict) else type(obj)}"
        )

    cleaned = {}
    for k, v in state_dict.items():
        if k.startswith("module."):
            cleaned[k[7:]] = v
        else:
            cleaned[k] = v

    msg = model.load_state_dict(cleaned, strict=False)
    print("[INFO] load_state_dict missing_keys:", msg.missing_keys)
    print("[INFO] load_state_dict unexpected_keys:", msg.unexpected_keys)
    return model


# =========================================================
# Constants / paths
# =========================================================
CLASS_NAMES = ["NORM", "MI", "STTC", "CD", "HYP"]
MI_INDEX = CLASS_NAMES.index("MI")

DATA_DIR = PROJECT_ROOT / "data" / "processed_v3"
GEORGIA_DIR = PROJECT_ROOT / "data" / "external_processed" / "georgia_5class"
RESULT_ROOT = PROJECT_ROOT / "results" / "external_eval" / "georgia" / "MCKI"
RESULT_ROOT.mkdir(parents=True, exist_ok=True)

DEFAULT_PRETRAINED_PRIMARY = PROJECT_ROOT / "checkpoints" / "refine_MCKI_Full.pth"
DEFAULT_PRETRAINED_FALLBACK = PROJECT_ROOT / "checkpoints" / "best_MCKI_pro_v4.pth"
DEFAULT_HARD_PAIRS = PROJECT_ROOT / "results" / "hard_pairs.csv"

STAGE8_EVAL_CFG = {
    "batch_size": 64,
    "num_workers": 4,
    "lp_epochs": 25,
    "finetune_epochs": 40,
    "ft_lr": 1e-4,
    "ft_weight_decay": 1e-5,
    "backbone_lr_mult": 0.1,
    "head_lr_mult": 1.0,
    "lp_lr": 1e-3,
    "lp_weight_decay": 1e-4,
    "monitor_metric": "Macro_AUPRC",
    "early_stop_patience": 8,
    "scheduler": "cosine",
    "tune_thresholds": True,
    "use_pos_weight": True,
}

FEWSHOT_RATIOS = {
    "Few_Shot_1%": 0.01,
    "Few_Shot_10%": 0.10,
    "Few_Shot_25%": 0.25,
}

ALL_PROTOCOLS = [
    "Linear_Probing",
    "Few_Shot_1%",
    "Few_Shot_10%",
    "Few_Shot_25%",
    "Full_Finetune",
]

SUMMARY_KEYS = [
    "Macro_AUC",
    "Macro_AUPRC",
    "Macro_F1",
    "MI_F1",
    "HNDR_Pair",
    "HNDR_Inst",
]


# =========================================================
# Utilities
# =========================================================
def normalize_protocol_name(protocol: str) -> str:
    alias = {
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


def resolve_default_pretrained() -> Path:
    if DEFAULT_PRETRAINED_PRIMARY.exists():
        return DEFAULT_PRETRAINED_PRIMARY
    return DEFAULT_PRETRAINED_FALLBACK


def seed_everything(seed: int):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def safe_float(x):
    try:
        return float(x)
    except Exception:
        return math.nan


def build_loader(dataset, batch_size, shuffle, num_workers, seed=None):
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


def compute_pos_weight_from_loader(loader, device):
    ys = []
    for _, y in loader:
        ys.append(y.float())
    y = torch.cat(ys, dim=0)
    pos = y.sum(dim=0)
    neg = y.shape[0] - pos
    pos_weight = (neg / (pos + 1e-6)).clamp(min=1.0, max=20.0)
    return pos_weight.to(device)


def build_scheduler(optimizer, total_epochs: int):
    return optim.lr_scheduler.CosineAnnealingLR(
        optimizer,
        T_max=max(total_epochs, 1),
        eta_min=1e-6,
    )


def build_optimizer_for_protocol(model, protocol, cfg):
    protocol = normalize_protocol_name(protocol)

    if protocol == "Linear_Probing":
        trainable_params = [p for p in model.parameters() if p.requires_grad]
        return optim.AdamW(
            trainable_params,
            lr=float(cfg.get("lp_lr", 1e-3)),
            weight_decay=float(cfg.get("lp_weight_decay", 1e-4)),
        )

    if protocol in FEWSHOT_RATIOS or protocol == "Full_Finetune":
        ft_lr = float(cfg.get("ft_lr", 1e-4))
        ft_wd = float(cfg.get("ft_weight_decay", 1e-5))
        backbone_lr_mult = float(cfg.get("backbone_lr_mult", 0.1))
        head_lr_mult = float(cfg.get("head_lr_mult", 1.0))

        backbone_params = []
        head_params = []
        for name, p in model.named_parameters():
            if not p.requires_grad:
                continue
            if "cls_head" in name:
                head_params.append(p)
            else:
                backbone_params.append(p)

        param_groups = []
        if backbone_params:
            param_groups.append({
                "params": backbone_params,
                "lr": ft_lr * backbone_lr_mult,
                "weight_decay": ft_wd,
            })
        if head_params:
            param_groups.append({
                "params": head_params,
                "lr": ft_lr * head_lr_mult,
                "weight_decay": ft_wd,
            })

        return optim.AdamW(param_groups)

    raise ValueError(protocol)


# =========================================================
# Dataset
# =========================================================
class InternalNPYDataset(Dataset):
    def __init__(self, x_path, y_path):
        self.x = np.load(x_path, mmap_mode="r")
        self.y = np.load(y_path, mmap_mode="r")

    def __len__(self):
        return len(self.y)

    def __getitem__(self, idx):
        x = torch.tensor(self.x[idx], dtype=torch.float32)  # (1000, 12)
        y = torch.tensor(self.y[idx], dtype=torch.float32)  # (5,)
        x = x.permute(1, 0)  # -> (12, 1000)
        return x, y


class ExternalNPYDataset(Dataset):
    def __init__(self, x_path, y_path):
        self.x = np.load(x_path, mmap_mode="r")
        self.y = np.load(y_path, mmap_mode="r")

    def __len__(self):
        return len(self.y)

    def __getitem__(self, idx):
        x = torch.tensor(self.x[idx], dtype=torch.float32)  # (1000, 12)
        y = torch.tensor(self.y[idx], dtype=torch.float32)  # (5,)
        x = x.permute(1, 0)  # -> (12, 1000)
        return x, y


# =========================================================
# Metrics / thresholds / HNDR
# =========================================================
def calculate_hndr(probs, targets, class_names, csv_path):
    if not os.path.exists(csv_path):
        return None, None, {}

    import pandas as pd
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
        correct_hard_samples += np.sum(binary_preds == binary_targets)

    if not pair_metrics:
        return None, None, {}

    hndr_pair = float(np.mean(list(pair_metrics.values())))
    hndr_inst = float(correct_hard_samples / total_hard_samples) if total_hard_samples > 0 else 0.0
    return hndr_pair, hndr_inst, pair_metrics


def find_best_thresholds(val_probs, val_targets, num_classes, grid=None):
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


def compute_metrics_raw(probs, targets):
    return {
        "Macro_AUC": float(roc_auc_score(targets, probs, average="macro")),
        "Macro_AUPRC": float(average_precision_score(targets, probs, average="macro")),
        "Macro_F1@0.5": float(f1_score(targets, (probs > 0.5).astype(int), average="macro", zero_division=0)),
        "MI_F1@0.5": float(f1_score(targets[:, MI_INDEX], (probs[:, MI_INDEX] > 0.5).astype(int), zero_division=0)),
    }


def compute_metrics_with_thresholds(probs, targets, thresholds):
    preds = (probs >= thresholds.reshape(1, -1)).astype(int)
    return {
        "Macro_F1": float(f1_score(targets, preds, average="macro", zero_division=0)),
        "MI_F1": float(f1_score(targets[:, MI_INDEX], preds[:, MI_INDEX], zero_division=0)),
        "Preds": preds,
    }


# =========================================================
# Dataloaders
# =========================================================
def build_internal_loaders(protocol, seed, batch_size, num_workers):
    protocol = normalize_protocol_name(protocol)

    train_x = DATA_DIR / "X_train.npy"
    val_x = DATA_DIR / "X_val.npy"
    train_y = DATA_DIR / "y_train_mh.npy"
    val_y = DATA_DIR / "y_val_mh.npy"

    train_ds = InternalNPYDataset(train_x, train_y)
    val_ds = InternalNPYDataset(val_x, val_y)

    if protocol in FEWSHOT_RATIOS:
        rng = np.random.default_rng(seed + 2026)
        n = len(train_ds)
        ratio = FEWSHOT_RATIOS[protocol]
        k = max(1, int(round(n * ratio)))
        idx = sorted(rng.choice(n, size=k, replace=False).tolist())
        train_ds = Subset(train_ds, idx)
        print(f"[INFO] {protocol}: using {k}/{n} training samples")
    elif protocol == "Full_Finetune":
        print(f"[INFO] Full_Finetune: using full training set {len(train_ds)}/{len(train_ds)}")

    train_loader = build_loader(
        train_ds,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        seed=seed + 23,
    )
    val_loader = build_loader(
        val_ds,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
    )
    return train_loader, val_loader


def build_external_loader(batch_size, num_workers):
    x_path = GEORGIA_DIR / "X_test.npy"
    y_path = GEORGIA_DIR / "y_test_mh.npy"
    ds = ExternalNPYDataset(x_path, y_path)
    return DataLoader(
        ds,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=torch.cuda.is_available(),
    )


# =========================================================
# Model / training / eval
# =========================================================
def build_model(protocol, pretrained_ckpt, device):
    protocol = normalize_protocol_name(protocol)

    model = TwoStageModel(num_classes=5).to(device)
    load_checkpoint_robust(model, str(pretrained_ckpt), device)

    if protocol == "Linear_Probing":
        for name, p in model.named_parameters():
            if "cls_head" in name:
                p.requires_grad = True
            else:
                p.requires_grad = False
        print("[INFO] Protocol = Linear_Probing (backbone frozen)")
    elif protocol in FEWSHOT_RATIOS or protocol == "Full_Finetune":
        for p in model.parameters():
            p.requires_grad = True
        print(f"[INFO] Protocol = {protocol} (all trainable)")
    else:
        raise ValueError(f"Unsupported protocol: {protocol}")

    return model


def forward_logits(model, x):
    return model.forward_cls(x)


def eval_loader(model, loader, device):
    model.eval()
    all_probs, all_targets = [], []

    with torch.no_grad():
        for x, y in loader:
            x = x.to(device, non_blocking=True)
            y = y.to(device, non_blocking=True)
            logits = forward_logits(model, x)
            probs = torch.sigmoid(logits).cpu().numpy()
            all_probs.append(probs)
            all_targets.append(y.cpu().numpy())

    probs = np.vstack(all_probs)
    targets = np.vstack(all_targets)
    return probs, targets


def train_one_epoch(model, loader, optimizer, criterion, device):
    model.train()
    total_loss = 0.0
    total_n = 0

    for x, y in loader:
        x = x.to(device, non_blocking=True)
        y = y.to(device, non_blocking=True)

        optimizer.zero_grad()
        logits = forward_logits(model, x)
        loss = criterion(logits, y)
        loss.backward()
        optimizer.step()

        bs = y.shape[0]
        total_loss += float(loss.item()) * bs
        total_n += bs

    return total_loss / max(total_n, 1)


def fit_protocol(protocol, seed, pretrained_ckpt, batch_size, num_workers, device, force_retrain=False):
    protocol = normalize_protocol_name(protocol)

    artifact_dir = RESULT_ROOT / "artifacts"
    artifact_dir.mkdir(parents=True, exist_ok=True)
    artifact_path = artifact_dir / f"{protocol}_seed{seed}.pth"

    model = build_model(protocol, pretrained_ckpt, device)

    if artifact_path.exists() and (not force_retrain):
        print(f"[INFO] Reusing trained protocol artifact: {artifact_path}")
        obj = torch.load(artifact_path, map_location="cpu", weights_only=False)
        model.load_state_dict(obj["model_state_dict"], strict=True)
        thresholds = np.asarray(obj["thresholds"], dtype=np.float32)
        return model, thresholds, artifact_path

    cfg = copy.deepcopy(STAGE8_EVAL_CFG)
    batch_size = int(batch_size or cfg["batch_size"])
    num_workers = int(num_workers if num_workers is not None else cfg["num_workers"])

    train_loader, val_loader = build_internal_loaders(protocol, seed, batch_size, num_workers)

    max_epochs = int(cfg["lp_epochs"] if protocol == "Linear_Probing" else cfg["finetune_epochs"])
    patience = int(cfg["early_stop_patience"])
    optimizer = build_optimizer_for_protocol(model, protocol, cfg)
    pos_weight = compute_pos_weight_from_loader(train_loader, device) if cfg.get("use_pos_weight", True) else None
    criterion = nn.BCEWithLogitsLoss(pos_weight=pos_weight)
    scheduler = build_scheduler(optimizer, max_epochs)

    best_val_score = -1.0
    best_state = None
    bad_epochs = 0

    print(f"[INFO] Start fitting protocol={protocol}")
    for epoch in range(1, max_epochs + 1):
        train_loss = train_one_epoch(model, train_loader, optimizer, criterion, device)
        scheduler.step()

        val_probs, val_targets = eval_loader(model, val_loader, device)
        val_auprc = average_precision_score(val_targets, val_probs, average="macro")

        print(f"[Epoch {epoch:02d}/{max_epochs}] train_loss={train_loss:.6f}  val_macro_auprc={val_auprc:.6f}")

        if val_auprc > best_val_score:
            best_val_score = val_auprc
            best_state = copy.deepcopy(model.state_dict())
            bad_epochs = 0
        else:
            bad_epochs += 1

        if bad_epochs >= patience:
            print(f"[INFO] Early stopping triggered (patience={patience})")
            break

    model.load_state_dict(best_state, strict=True)
    val_probs, val_targets = eval_loader(model, val_loader, device)
    thresholds, per_class_val_f1 = find_best_thresholds(val_probs, val_targets, len(CLASS_NAMES))

    obj = {
        "protocol": protocol,
        "seed": seed,
        "thresholds": thresholds.astype(np.float32),
        "per_class_val_f1": per_class_val_f1.astype(np.float32),
        "best_val_auprc": float(best_val_score),
        "model_state_dict": model.state_dict(),
        "class_names": CLASS_NAMES,
        "stage8_aligned_cfg": cfg,
    }
    torch.save(obj, artifact_path)
    print(f"[OK] Saved protocol artifact -> {artifact_path}")
    return model, thresholds, artifact_path


# =========================================================
# Save / summarize helpers
# =========================================================
def save_json(obj, path):
    with open(path, "w", encoding="utf-8") as f:
        json.dump(obj, f, ensure_ascii=False, indent=2)


def format_mean_std(mean, std):
    return f"{mean:.4f} ± {std:.4f}"


def load_run_metrics(protocol, seed):
    protocol = normalize_protocol_name(protocol)
    run_dir = RESULT_ROOT / f"{protocol}_seed{seed}"
    metrics_path = run_dir / "metrics.json"
    if not metrics_path.exists():
        raise FileNotFoundError(f"Missing metrics file: {metrics_path}")
    with open(metrics_path, "r", encoding="utf-8") as f:
        return json.load(f)


def summarize_protocol(protocol, seeds):
    rows = []
    for seed in seeds:
        obj = load_run_metrics(protocol, seed)
        rows.append(obj)

    summary = {"Protocol": normalize_protocol_name(protocol), "Seeds": [int(s) for s in seeds], "Num_Seeds": len(rows)}
    for k in SUMMARY_KEYS:
        vals = np.array([safe_float(r.get(k, math.nan)) for r in rows], dtype=float)
        summary[k] = {
            "mean": float(np.nanmean(vals)),
            "std": float(np.nanstd(vals, ddof=0)),
            "values": [float(v) for v in vals],
        }
    return summary, rows


def save_summary_json(all_summary, path):
    save_json(all_summary, path)


def save_summary_csv(all_summary, path):
    fieldnames = ["Protocol"] + SUMMARY_KEYS
    with open(path, "w", newline="", encoding="utf-8-sig") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for protocol in ALL_PROTOCOLS:
            s = all_summary["protocol_summaries"][protocol]
            row = {"Protocol": protocol}
            for k in SUMMARY_KEYS:
                row[k] = format_mean_std(s[k]["mean"], s[k]["std"])
            writer.writerow(row)


def save_summary_md(all_summary, path):
    lines = []
    lines.append("# Georgia External MCKI Summary")
    lines.append("")
    lines.append(f"- Result root: `{all_summary['result_root']}`")
    lines.append(f"- Seeds: `{all_summary['seeds']}`")
    lines.append("")

    headers = ["Protocol"] + SUMMARY_KEYS
    lines.append("| " + " | ".join(headers) + " |")
    lines.append("|" + "|".join(["---"] * len(headers)) + "|")

    for protocol in ALL_PROTOCOLS:
        s = all_summary["protocol_summaries"][protocol]
        row = [protocol]
        for k in SUMMARY_KEYS:
            row.append(format_mean_std(s[k]["mean"], s[k]["std"]))
        lines.append("| " + " | ".join(row) + " |")

    lines.append("")
    lines.append("## Per-seed raw results")
    lines.append("")

    for protocol in ALL_PROTOCOLS:
        lines.append(f"### {protocol}")
        lines.append("")
        lines.append("| Seed | " + " | ".join(SUMMARY_KEYS) + " |")
        lines.append("|" + "|".join(["---"] * (len(SUMMARY_KEYS) + 1)) + "|")
        for r in all_summary["raw_results"][protocol]:
            row = [str(r["Seed"])] + [f"{safe_float(r.get(k, math.nan)):.6f}" for k in SUMMARY_KEYS]
            lines.append("| " + " | ".join(row) + " |")
        lines.append("")

    path.write_text("\n".join(lines), encoding="utf-8")


def summarize_all(seeds):
    all_summary = {
        "result_root": str(RESULT_ROOT),
        "seeds": [int(s) for s in seeds],
        "protocol_summaries": {},
        "raw_results": {},
    }

    for protocol in ALL_PROTOCOLS:
        summary, rows = summarize_protocol(protocol, seeds)
        all_summary["protocol_summaries"][protocol] = summary
        all_summary["raw_results"][protocol] = rows

    json_path = RESULT_ROOT / "georgia_external_MCKI_summary.json"
    csv_path = RESULT_ROOT / "georgia_external_MCKI_summary.csv"
    md_path = RESULT_ROOT / "georgia_external_MCKI_summary.md"

    save_summary_json(all_summary, json_path)
    save_summary_csv(all_summary, csv_path)
    save_summary_md(all_summary, md_path)

    print("\n" + "=" * 100)
    print("Georgia External MCKI Summary")
    print("=" * 100)
    for protocol in ALL_PROTOCOLS:
        s = all_summary["protocol_summaries"][protocol]
        print(f"\n[Protocol] {protocol}")
        for k in SUMMARY_KEYS:
            print(f"  {k}: {s[k]['mean']:.4f} ± {s[k]['std']:.4f}")

    print("\n[OK] Saved:")
    print(f"  JSON -> {json_path}")
    print(f"  CSV  -> {csv_path}")
    print(f"  MD   -> {md_path}")


def run_single(protocol, seed, batch_size, num_workers, pretrained_ckpt, hard_pairs_csv, force_retrain):
    protocol = normalize_protocol_name(protocol)
    seed_everything(seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    print("[INFO] device =", device)
    print("[INFO] protocol =", protocol)
    print("[INFO] seed =", seed)
    print("[INFO] pretrained_ckpt =", pretrained_ckpt)
    print("[INFO] hard_pairs_csv =", hard_pairs_csv)

    model, thresholds, artifact_path = fit_protocol(
        protocol=protocol,
        seed=seed,
        pretrained_ckpt=pretrained_ckpt,
        batch_size=batch_size,
        num_workers=num_workers,
        device=device,
        force_retrain=force_retrain,
    )

    ext_loader = build_external_loader(batch_size, num_workers)
    probs, targets = eval_loader(model, ext_loader, device)

    raw_metrics = compute_metrics_raw(probs, targets)
    th_metrics = compute_metrics_with_thresholds(probs, targets, thresholds)
    hndr_pair, hndr_inst, pair_metrics = calculate_hndr(
        probs=probs,
        targets=targets,
        class_names=CLASS_NAMES,
        csv_path=hard_pairs_csv,
    )

    final_metrics = {
        "Protocol": protocol,
        "Seed": int(seed),
        "Macro_AUC": float(raw_metrics["Macro_AUC"]),
        "Macro_AUPRC": float(raw_metrics["Macro_AUPRC"]),
        "Macro_F1": float(th_metrics["Macro_F1"]),
        "MI_F1": float(th_metrics["MI_F1"]),
        "Macro_F1@0.5": float(raw_metrics["Macro_F1@0.5"]),
        "MI_F1@0.5": float(raw_metrics["MI_F1@0.5"]),
        "HNDR_Pair": None if hndr_pair is None else float(hndr_pair),
        "HNDR_Inst": None if hndr_inst is None else float(hndr_inst),
        "Artifact_Path": str(artifact_path),
        "Hard_Pairs_CSV": str(hard_pairs_csv),
    }

    run_dir = RESULT_ROOT / f"{protocol}_seed{seed}"
    run_dir.mkdir(parents=True, exist_ok=True)

    np.save(run_dir / "external_probs.npy", probs)
    np.save(run_dir / "external_targets.npy", targets)
    np.save(run_dir / "thresholds.npy", thresholds.astype(np.float32))
    save_json(final_metrics, run_dir / "metrics.json")
    save_json({k: float(v) for k, v in pair_metrics.items()}, run_dir / "pair_metrics.json")

    print("\n=== Georgia External MCKI Results ===")
    for k, v in final_metrics.items():
        if isinstance(v, float):
            print(f"{k}: {v:.6f}")
        else:
            print(f"{k}: {v}")

    print(f"\n[OK] Saved results -> {run_dir}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--protocol",
        type=str,
        default="Linear_Probing",
        choices=ALL_PROTOCOLS + ["ALL", "Full_Finetuning", "Full_Fineline", "full_fineline"],
    )
    parser.add_argument("--seed", type=int, default=42, help="Single-seed mode only")
    parser.add_argument("--seeds", type=int, nargs="+", default=[42, 123, 1024], help="Multi-seed mode / summary mode")
    parser.add_argument("--batch-size", type=int, default=STAGE8_EVAL_CFG["batch_size"])
    parser.add_argument("--num-workers", type=int, default=STAGE8_EVAL_CFG["num_workers"])
    parser.add_argument("--pretrained-ckpt", type=str, default=str(resolve_default_pretrained()))
    parser.add_argument("--hard-pairs-csv", type=str, default=str(DEFAULT_HARD_PAIRS))
    parser.add_argument("--force-retrain", action="store_true")
    parser.add_argument("--summarize-only", action="store_true")
    args = parser.parse_args()

    args.protocol = normalize_protocol_name(args.protocol)

    if args.summarize_only:
        summarize_all(args.seeds)
        return

    if args.protocol == "ALL":
        for seed in args.seeds:
            for protocol in ALL_PROTOCOLS:
                print("\n" + "=" * 100)
                print(f"Running protocol={protocol}, seed={seed}")
                print("=" * 100)
                run_single(
                    protocol=protocol,
                    seed=seed,
                    batch_size=args.batch_size,
                    num_workers=args.num_workers,
                    pretrained_ckpt=args.pretrained_ckpt,
                    hard_pairs_csv=args.hard_pairs_csv,
                    force_retrain=args.force_retrain,
                )
        summarize_all(args.seeds)
        return

    run_single(
        protocol=args.protocol,
        seed=args.seed,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        pretrained_ckpt=args.pretrained_ckpt,
        hard_pairs_csv=args.hard_pairs_csv,
        force_retrain=args.force_retrain,
    )


if __name__ == "__main__":
    main()