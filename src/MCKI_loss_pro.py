import torch
import torch.nn as nn
import torch.nn.functional as F


DEFAULT_S_HIERARCHY = [
    [1.0000, 0.0001, 0.0020, 0.0297, 0.0002],
    [0.0001, 1.0000, 0.1405, 0.2093, 0.1111],
    [0.0020, 0.1405, 1.0000, 0.1184, 0.2404],
    [0.0297, 0.2093, 0.1184, 1.0000, 0.1157],
    [0.0002, 0.1111, 0.2404, 0.1157, 1.0000],
]


class MCKILossPro(nn.Module):
    """
    Phase-1 MCKI:
      - multi-label aware
      - explicit exclusion of label-overlap samples from hard negatives
      - relation matrix externally injectable (defaults to S_hierarchy)
      - supports two-view contrastive pretraining by passing features_view2
    """

    def __init__(
        self,
        temperature: float = 0.07,
        alpha: float = 2.0,
        hard_negative_threshold: float = 0.10,
        use_continuous_weights: bool = False,
        relation_matrix=None,
        eps: float = 1e-8,
    ):
        super().__init__()
        self.temp = float(temperature)
        self.alpha = float(alpha)
        self.hard_negative_threshold = float(hard_negative_threshold)
        self.use_continuous_weights = bool(use_continuous_weights)
        self.eps = float(eps)

        base_S = relation_matrix if relation_matrix is not None else DEFAULT_S_HIERARCHY
        self.register_buffer('S', torch.tensor(base_S, dtype=torch.float32))

    def set_relation_matrix(self, relation_matrix) -> None:
        self.S = torch.as_tensor(relation_matrix, dtype=torch.float32, device=self.S.device)

    def _prepare_labels(self, labels: torch.Tensor) -> torch.Tensor:
        if labels.ndim == 1:
            labels = F.one_hot(labels.long(), num_classes=self.S.shape[0]).float()
        return labels.float()

    def _pair_overlap(self, labels: torch.Tensor) -> torch.Tensor:
        return torch.matmul(labels, labels.T)

    def _pair_jaccard(self, labels: torch.Tensor) -> torch.Tensor:
        inter = self._pair_overlap(labels)
        counts = labels.sum(dim=1, keepdim=True)
        union = counts + counts.T - inter
        return inter / union.clamp_min(self.eps)

    def _relation_scores(self, labels: torch.Tensor) -> torch.Tensor:
        relation = torch.matmul(torch.matmul(labels, self.S), labels.T)
        counts = labels.sum(dim=1, keepdim=True)
        norm = counts * counts.T
        relation = relation / norm.clamp_min(1.0)
        return relation.clamp(min=0.0)

    def forward(
        self,
        features: torch.Tensor,
        labels: torch.Tensor,
        features_view2: torch.Tensor = None,
        labels_view2: torch.Tensor = None,
    ) -> torch.Tensor:
        device = features.device
        S = self.S.to(device)
        if S.data_ptr() != self.S.data_ptr():
            self.S = S

        labels = self._prepare_labels(labels).to(device)
        features = F.normalize(features, dim=1)

        if features_view2 is not None:
            features_view2 = F.normalize(features_view2, dim=1)
            if labels_view2 is None:
                labels_view2 = labels
            else:
                labels_view2 = self._prepare_labels(labels_view2).to(device)
            features = torch.cat([features, features_view2], dim=0)
            labels = torch.cat([labels, labels_view2], dim=0)

        logits = torch.matmul(features, features.T) / self.temp
        eye = torch.eye(logits.shape[0], device=device, dtype=torch.bool)

        overlap = self._pair_overlap(labels)
        jaccard = self._pair_jaccard(labels)
        relation_scores = self._relation_scores(labels)

        positive_mask = (jaccard >= 0.5).float()
        positive_mask = positive_mask.masked_fill(eye, 0.0)
        positive_weights = (jaccard * positive_mask).detach()

        neutral_mask = ((jaccard > 0.0) & (jaccard < 0.5)).float()
        neutral_mask = neutral_mask.masked_fill(eye, 0.0)

        hard_negative_mask = (jaccard <= 0.0).float() * (~eye).float() * (
                    relation_scores > self.hard_negative_threshold).float()
        penalty_weights = torch.ones_like(logits)
        if self.use_continuous_weights:
            penalty_weights = penalty_weights + hard_negative_mask * (self.alpha * relation_scores)
        else:
            penalty_weights = penalty_weights + hard_negative_mask * self.alpha

        logits_max, _ = torch.max(logits, dim=1, keepdim=True)
        logits_sub = logits - logits_max.detach()

        valid_mask = (~eye).float()
        exp_logits = torch.exp(logits_sub) * valid_mask * penalty_weights
        log_prob = logits_sub - torch.log(exp_logits.sum(dim=1, keepdim=True) + self.eps)

        pos_denom = positive_weights.sum(dim=1)
        per_sample = -(positive_weights * log_prob).sum(dim=1) / pos_denom.clamp_min(self.eps)
        valid_rows = pos_denom > 0
        if valid_rows.any():
            return per_sample[valid_rows].mean()
        return -log_prob.mean()
