import json
import os
from typing import Iterable, Optional

import numpy as np

CLASS_NAMES = ['NORM', 'MI', 'STTC', 'CD', 'HYP']
DEFAULT_S_HIERARCHY = np.array([
    [1.0000, 0.0001, 0.0020, 0.0297, 0.0002],
    [0.0001, 1.0000, 0.1405, 0.2093, 0.1111],
    [0.0020, 0.1405, 1.0000, 0.1184, 0.2404],
    [0.0297, 0.2093, 0.1184, 1.0000, 0.1157],
    [0.0002, 0.1111, 0.2404, 0.1157, 1.0000],
], dtype=np.float32)


def load_prior_matrix(relation_matrix_values: Optional[Iterable[Iterable[float]]] = None) -> np.ndarray:
    if relation_matrix_values is None:
        return DEFAULT_S_HIERARCHY.copy()
    arr = np.asarray(relation_matrix_values, dtype=np.float32)
    if arr.shape != (len(CLASS_NAMES), len(CLASS_NAMES)):
        raise ValueError(f"relation_matrix_values shape must be {(len(CLASS_NAMES), len(CLASS_NAMES))}, got {arr.shape}")
    arr = 0.5 * (arr + arr.T)
    np.fill_diagonal(arr, 1.0)
    return arr


def estimate_confusion_matrix_from_probs(probs: np.ndarray, targets: np.ndarray, eps: float = 1e-8) -> np.ndarray:
    """
    Estimate pairwise diagnostic confusion from multi-label probabilities.
    For each pair (i, j), measure how much class j activates on samples that are
    positive for i and negative for j, and vice versa. Then symmetrize.
    """
    if probs.shape != targets.shape:
        raise ValueError(f"probs and targets must have same shape, got {probs.shape} vs {targets.shape}")

    num_classes = probs.shape[1]
    conf = np.zeros((num_classes, num_classes), dtype=np.float32)
    np.fill_diagonal(conf, 1.0)

    for i in range(num_classes):
        for j in range(i + 1, num_classes):
            mask_i = (targets[:, i] >= 0.5) & (targets[:, j] < 0.5)
            mask_j = (targets[:, j] >= 0.5) & (targets[:, i] < 0.5)
            s_i_to_j = float(probs[mask_i, j].mean()) if mask_i.any() else 0.0
            s_j_to_i = float(probs[mask_j, i].mean()) if mask_j.any() else 0.0
            score = 0.5 * (s_i_to_j + s_j_to_i)
            conf[i, j] = score
            conf[j, i] = score

    off_diag = conf.copy()
    np.fill_diagonal(off_diag, 0.0)
    max_off = float(off_diag.max())
    if max_off > eps:
        off_diag /= max_off
    conf = off_diag
    np.fill_diagonal(conf, 1.0)
    return conf.astype(np.float32)


def blend_relation_matrices(
    prior: np.ndarray,
    confusion: np.ndarray,
    lambda_prior: float = 0.7,
    lambda_conf: float = 0.3,
) -> np.ndarray:
    if prior.shape != confusion.shape:
        raise ValueError(f"prior/confusion shape mismatch: {prior.shape} vs {confusion.shape}")
    total = float(lambda_prior + lambda_conf)
    if total <= 0:
        raise ValueError("lambda_prior + lambda_conf must be positive")
    lambda_prior /= total
    lambda_conf /= total

    hybrid = lambda_prior * prior + lambda_conf * confusion
    hybrid = 0.5 * (hybrid + hybrid.T)
    np.fill_diagonal(hybrid, 1.0)
    return hybrid.astype(np.float32)


def save_relation_artifacts(save_dir: str, prior: np.ndarray, confusion: np.ndarray, hybrid: np.ndarray) -> None:
    os.makedirs(save_dir, exist_ok=True)
    np.save(os.path.join(save_dir, 'S_prior.npy'), prior)
    np.save(os.path.join(save_dir, 'S_confusion.npy'), confusion)
    np.save(os.path.join(save_dir, 'S_hybrid.npy'), hybrid)

    payload = {
        'class_names': CLASS_NAMES,
        'S_prior': prior.tolist(),
        'S_confusion': confusion.tolist(),
        'S_hybrid': hybrid.tolist(),
    }
    with open(os.path.join(save_dir, 'relation_matrices.json'), 'w', encoding='utf-8') as f:
        json.dump(payload, f, ensure_ascii=False, indent=2)
