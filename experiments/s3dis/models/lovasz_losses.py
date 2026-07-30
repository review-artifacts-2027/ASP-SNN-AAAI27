import torch
import torch.nn as nn
import torch.nn.functional as F


def _lovasz_grad(gt_sorted: torch.Tensor) -> torch.Tensor:
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

    C = probas.shape[-1]
    probas = probas.reshape(-1, C)
    labels = labels.reshape(-1)
    if ignore is None:
        return probas, labels
    valid = labels != ignore
    return probas[valid], labels[valid]


def lovasz_softmax_flat(probas: torch.Tensor, labels: torch.Tensor,
                        classes: str = "present") -> torch.Tensor:

    if probas.numel() == 0:
        return probas * 0.0

    C = probas.shape[-1]
    losses = []
    class_iter = list(range(C)) if classes == "all" else \
                 labels.unique().tolist()

    for c in class_iter:
        fg = (labels == c).float()
        if classes == "present" and fg.sum() == 0:
            continue

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
    def __init__(self, ignore_index: int = -1, classes: str = "present"):
        super().__init__()
        self.ignore_index = ignore_index
        self.classes = classes

    def forward(self, logits: torch.Tensor,
                labels: torch.Tensor) -> torch.Tensor:

        probas = F.softmax(logits.float(), dim=-1)
        probas_flat, labels_flat = _flatten_probas(
            probas, labels, ignore=self.ignore_index,
        )
        return lovasz_softmax_flat(
            probas_flat, labels_flat, classes=self.classes,
        )
