import torch
import torch.nn as nn
import torch.nn.functional as F
from tqdm import tqdm
from utils.utils import metrics
from utils.dice_score import dice_coeff
from sklearn.metrics import accuracy_score, recall_score, f1_score, precision_score


@torch.inference_mode()
def evaluate(net, dataloader, device, amp):
    net.eval()
    num_val_batches = len(dataloader)
    dice_score = 0
    accuracy = 0
    precision = 0
    recall = 0
    f1 = 0
    loss = 0
    preds = []
    targets = []
    # iterate over the validation set
    with torch.autocast(device.type if device.type != 'mps' else 'cpu', enabled=amp):
        for batch in tqdm(dataloader, total=num_val_batches, desc='Validation round', unit='batch', leave=False):
            image, mask_true, label_true = batch['image'], batch['mask'], batch['label']
            # move images and labels to correct device and type
            image = image.to(device=device, dtype=torch.float32, memory_format=torch.channels_last)
            mask_true = mask_true.to(device=device, dtype=torch.long)
            label_true = label_true.to(device=device, dtype=torch.float)
            # predict the mask and label
            mask_pred, label_pred = net(image)['mask'], net(image)['label']
            assert mask_true.min() >= 0 and mask_true.max() <= 1, 'True mask indices should be in [0, 1]'
            # mask threshold
            mask_pred = (torch.sigmoid(mask_pred) >= 0.5).float()
            # compute validation loss
            # label_loss = F.binary_cross_entropy_with_logits(label_preds[0], label_true.float()) + \
            #              F.binary_cross_entropy_with_logits(label_preds[1], label_true.float()) + \
            #              0.1 * F.binary_cross_entropy_with_logits(label_preds[2], label_true.float())
            label_loss = F.binary_cross_entropy_with_logits(label_pred, label_true)
            loss += label_loss
            # label unsmoothing
            label_true = (label_true > 0.5).float()
            targets.append(label_true.cpu())
            # label_pred = label_preds[0] + label_preds[1] + 0.1 * label_preds[2]
            label_pred = torch.sigmoid(label_pred)
            preds.append(label_pred.cpu())
            # label threshold
            label_pred[:, :3] = (label_pred[:, :3] == torch.max(label_pred[:, :3], dim=1, keepdim=True)[0]).float()
            label_pred[:, 3:] = (label_pred[:, 3:] >= 0.5).float()
            # compute metrics by sklearn
            dice_score += dice_coeff(mask_pred.squeeze(1), mask_true, reduce_batch_first=False)
            accuracy += accuracy_score(label_true.cpu(), label_pred.cpu())
            precision += precision_score(label_true.cpu(), label_pred.cpu(), average='samples')
            recall += recall_score(label_true.cpu(), label_pred.cpu(), average='samples')
            f1 += f1_score(label_true.cpu(), label_pred.cpu(), average='samples')
        mAP_score, ap, OP, OR, OF1, CP, CR, CF1 = metrics(torch.cat(targets).numpy(), torch.cat(preds).numpy())

    net.train()

    return dice_score / num_val_batches, loss / num_val_batches, mAP_score, ap, OP, OR, OF1, CP, CR, CF1, \
           accuracy / num_val_batches, precision / num_val_batches, recall / num_val_batches, f1 / num_val_batches
