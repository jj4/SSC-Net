import torch
from torch.autograd import Variable
import numpy as np


def get_bounding_box(tensor):
    tensor = tensor.squeeze(1)
    batch, height, width = tensor.shape
    boxes = torch.zeros((batch, 4))
    for i in range(batch):
        indices = torch.nonzero(tensor[i])
        if indices.numel() == 0:
            boxes[i] = torch.tensor([0, 0, height, width])
        else:
            min_y, max_y = indices[:, 0].min(), indices[:, 0].max()
            min_x, max_x = indices[:, 1].min(), indices[:, 1].max()
            boxes[i] = torch.tensor([min_x - 1, min_y - 1, max_x + 1, max_y + 1])
    return to_varabile(boxes)


def to_varabile(tensor, requires_grad=False, is_cuda=True):
    if is_cuda:
        tensor = tensor.cuda()
    var = Variable(tensor, requires_grad=requires_grad)
    return var


def average_precision(output, target):
    epsilon = 1e-8
    # sort examples
    indices = output.argsort()[::-1]
    # Computes prec@i
    total_count_ = np.cumsum(np.ones((len(output), 1)))

    target_ = target[indices]
    ind = target_ == 1
    pos_count_ = np.cumsum(ind)
    total = pos_count_[-1]
    pos_count_[np.logical_not(ind)] = 0
    pp = pos_count_ / total_count_
    precision_at_i_ = np.sum(pp)
    precision_at_i = precision_at_i_ / (total + epsilon)

    return precision_at_i


def metrics(targs, preds):
    """Returns the model's average precision for each class
    Return:
        ap (FloatTensor): 1xK tensor, with avg precision for each class k
    """
    if np.size(preds) == 0:
        return 0
    n_class = preds.shape[1]
    preds_ = preds.copy()
    preds_[:, :3] = np.where(preds_[:, :3] == np.max(preds_[:, :3], axis=1).reshape(-1, 1), 1, 0)

    ap, Nc, Np, Ng = np.zeros(n_class), np.zeros(n_class), np.zeros(n_class), np.zeros(n_class)
    # compute average precision for each class
    for k in range(n_class):
        # sort scores
        scores = preds[:, k]
        scores_ = preds_[:, k]
        targets = targs[:, k]
        # compute average precision
        ap[k] = 100 * average_precision(scores, targets)
        # compute N_predict, N_correct, N_gt
        Ng[k] = np.sum(targets == 1)
        Np[k] = np.sum(scores_ >= 0.5)
        Nc[k] = np.sum(targets * (scores_ >= 0.5))
    # compute mean average precision
    map_ = ap.mean()
    Np[Np == 0] = 1  # avoid devided by 0
    # compute Overall P, R, F1
    OP = np.sum(Nc) / np.sum(Np)
    OR = np.sum(Nc) / np.sum(Ng)
    OF1 = (2 * OP * OR) / (OP + OR)
    # compute per-Category P, R, F1
    CP = np.sum(Nc / Np) / n_class
    CR = np.sum(Nc / Ng) / n_class
    CF1 = (2 * CP * CR) / (CP + CR)

    return map_, ap.round(2), OP, OR, OF1, CP, CR, CF1
