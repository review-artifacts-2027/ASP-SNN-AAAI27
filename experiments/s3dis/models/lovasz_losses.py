"""
models/lovasz_losses.py — Lovász-Softmax loss for point-cloud segmentation.

Differentiable surrogate for the IoU (Jaccard) index. Optimizes exactly
what mIoU measures, unlike weighted cross-entropy which optimizes accuracy.

Reference:
    Berman, Triki, Blaschko. "The Lovász-Softmax loss: A tractable
    surrogate for the optimization of the intersection-over-union measure
    in neural networks." CVPR 2018.

    Widely used in LiDAR point-cloud segmentation (Cylinder3D, RangeNet++,
    SPVNAS, LatticeNet, etc.) as a complement to CE for handling severe
    class imbalance and boundary regions.

Adapted from the reference implementation at
    https://github.com/bermanmaxim/LovaszSoftmax
and simplified for the multi-class point-cloud case (no image ops).
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


def _lovasz_grad(gt_sorted: torch.Tensor) -> torch.Tensor:
    """
    Gradient of the Lovász extension w.r.t. the sorted errors.

    Args:
        gt_sorted: [P] binary ground-truth sorted by descending prediction error

    Returns:
        jaccard: [P] same shape, Lovász gradient
    """
    p = gt_sorted.numel()
    gts = gt_sorted.sum()
    intersection = gts - gt_sorted.float().cumsum(0)
    union = gts + (1.0 - gt_sorted.float()).cumsum(0)
    jaccard = 1.0 - intersection / union
    if p > 1:
        jaccard[1:p] = jaccard[1:p] - jaccard[0:-1]
    return jaccard


def _flatten_probas(probas: torch.Tensor, labels: torch.Tensor,
                    ignore: int = None):
    """
    Flatten predictions and labels; optionally remove ignore_index points.

    Args:
        probas: [..., C]  softmax probabilities (any leading dims)
        labels: [...]     integer labels (matching leading dims)
        ignore: label value to exclude

    Returns:
        probas_flat: [P_valid, C]
        labels_flat: [P_valid]
    """
    C = probas.shape[-1]
    probas = probas.reshape(-1, C)
    labels = labels.reshape(-1)
    if ignore is None:
        return probas, labels
    valid = labels != ignore
    return probas[valid], labels[valid]


def lovasz_softmax_flat(probas: torch.Tensor, labels: torch.Tensor,
                        classes: str = "present") -> torch.Tensor:
    """
    Multi-class Lovász-Softmax loss on flattened tensors.

    Args:
        probas:  [P, C]  softmax probabilities
        labels:  [P]     integer labels in [0, C-1]
        classes: 'all' (average over all C) or 'present' (only classes in labels)

    Returns:
        loss: scalar
    """
    if probas.numel() == 0:
        return probas * 0.0

    C = probas.shape[-1]
    losses = []
    class_iter = list(range(C)) if classes == "all" else \
                 labels.unique().tolist()

    for c in class_iter:
        fg = (labels == c).float()               # [P] binary ground truth
        if classes == "present" and fg.sum() == 0:
            continue
        # Error for this class: how wrong the probability is for this fg mask
        errors = (fg - probas[:, c]).abs()
        errors_sorted, perm = torch.sort(errors, descending=True)
        fg_sorted = fg[perm]
        losses.append(
            torch.dot(errors_sorted, _lovasz_grad(fg_sorted))
        )

    if len(losses) == 0:
        return probas.sum() * 0.0
    return torch.stack(losses).mean()


class LovaszSoftmaxLoss(nn.Module):
    """
    Lovász-Softmax loss module.

    Standard usage: combine with cross-entropy as
        loss = ce_loss + lovasz_lambda * lovasz_softmax_loss(logits, labels)

    lovasz_lambda = 1.0 is the common default in LiDAR seg papers.
    """

    def __init__(self, ignore_index: int = -1, classes: str = "present"):
        super().__init__()
        self.ignore_index = ignore_index
        self.classes = classes

    def forward(self, logits: torch.Tensor,
                labels: torch.Tensor) -> torch.Tensor:
        """
        Args:
            logits: [..., C]  raw logits (any leading dims)
            labels: [...]     integer labels

        Returns:
            loss: scalar
        """
        # Cast to fp32 for numerical stability under AMP — the sort +
        # cumsum + division chain accumulates errors in fp16.
        probas = F.softmax(logits.float(), dim=-1)
        probas_flat, labels_flat = _flatten_probas(
            probas, labels, ignore=self.ignore_index,
        )
        return lovasz_softmax_flat(
            probas_flat, labels_flat, classes=self.classes,
        )