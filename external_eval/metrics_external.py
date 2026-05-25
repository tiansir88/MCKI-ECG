import numpy as np
from sklearn.metrics import roc_auc_score, average_precision_score, f1_score

MI_INDEX = 1

def compute_multilabel_metrics(y_true, y_prob, threshold=0.5):
    y_pred = (y_prob >= threshold).astype(np.int32)

    return {
        "Macro_AUC": float(roc_auc_score(y_true, y_prob, average="macro")),
        "Macro_AUPRC": float(average_precision_score(y_true, y_prob, average="macro")),
        "Macro_F1": float(f1_score(y_true, y_pred, average="macro", zero_division=0)),
        "MI_F1": float(f1_score(y_true[:, MI_INDEX], y_pred[:, MI_INDEX], zero_division=0)),
    }
