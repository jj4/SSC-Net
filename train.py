import argparse
import logging
import os
import torch
import torch.nn as nn
import torch.nn.functional as F
from pathlib import Path
from torch import optim
from torch.utils.data import DataLoader
from tqdm import tqdm

import wandb
from evaluate import evaluate
from model import SSCNet
from utils import BasicDataset, dice_loss
import warnings
warnings.filterwarnings('ignore')

dir_img = './data/imgs/'
dir_mask = './data/masks/'
dir_label = './data/labels/'
dir_checkpoint = './checkpoints/'

def train(
        model,
        device,
        epochs: int = 5,
        batch_size: int = 1,
        learning_rate: float = 1e-6,
        save_checkpoint: bool = True,
        img_scale: float = 1.0,
        amp: bool = True,
        weight_decay: float = 1e-8,
        momentum: float = 0.999,
        gradient_clipping: float = 1.0,
):
    # 1. Create dataset
    train_set = BasicDataset(dir_img + 'train', dir_mask + 'train', dir_label + 'train', img_scale)
    valid_set = BasicDataset(dir_img + 'valid', dir_mask + 'valid', dir_label + 'valid', img_scale, valid=True)

    # 2. Create data loaders
    loader_args = dict(batch_size=batch_size, num_workers=os.cpu_count(), pin_memory=True)
    train_loader = DataLoader(train_set, shuffle=True, **loader_args)
    valid_loader = DataLoader(valid_set, shuffle=False, drop_last=True, **loader_args)

    # (Initialize logging)
    experiment = wandb.init(project='SSCNet', resume='allow', anonymous='must')
    experiment.config.update(
        dict(epochs=epochs, batch_size=batch_size, learning_rate=learning_rate, save_checkpoint=save_checkpoint,
             img_scale=img_scale, amp=amp)
    )

    logging.info(f'''Starting training:
        Epochs:          {epochs}
        Batch size:      {batch_size}
        Learning rate:   {learning_rate}
        Training size:   {len(train_set)}
        Validation size: {len(valid_set)}
        Checkpoints:     {save_checkpoint}
        Device:          {device.type}
        Images scaling:  {img_scale}
        Mixed Precision: {amp}
    ''')

    # 3. Set up the optimizer, the loss, the learning rate scheduler and the loss scaling for AMP
    optimizer = optim.RMSprop(model.parameters(), lr=learning_rate, weight_decay=weight_decay, momentum=momentum)
    # scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, 'max', patience=5)  # goal: maximize Dice score
    grad_scaler = torch.cuda.amp.GradScaler(enabled=amp)
    criterion = nn.BCEWithLogitsLoss()
    global_step = 0

    # 4. Begin training
    for epoch in range(1, epochs + 1):
        model.train()
        epoch_loss, total_los, mask_los, label_los = 0, 0, 0, 0
        with tqdm(total=len(train_set), desc=f'Epoch {epoch}/{epochs}', unit='img') as pbar:
            for batch in train_loader:
                images, true_masks, true_labels = batch['image'], batch['mask'], batch['label']
                images = images.to(device=device, dtype=torch.float32, memory_format=torch.channels_last)
                true_masks = true_masks.to(device=device, dtype=torch.long)
                true_labels = true_labels.to(device=device, dtype=torch.float)

                with torch.autocast(device.type if device.type != 'mps' else 'cpu', enabled=amp):
                    masks_pred = model(images)['mask']
                    labels_pred = model(images)['label']

                    mask_loss = criterion(masks_pred.squeeze(1), true_masks.float())
                    mask_loss += dice_loss(F.sigmoid(masks_pred).squeeze(1), true_masks.float())
                    label_loss = criterion(labels_pred, true_labels.float())
                    # label_loss = criterion(labels_pred[0], true_labels.float()) + criterion(labels_pred[1], true_labels.float()) + 0.1 * criterion(labels_pred[2], true_labels.float())
                    loss = mask_loss + 0.5 * label_loss

                optimizer.zero_grad(set_to_none=True)
                grad_scaler.scale(loss).backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), gradient_clipping)
                grad_scaler.step(optimizer)
                grad_scaler.update()

                pbar.update(images.shape[0])
                global_step += 1
                # epoch_loss += loss.item() / len(train_loader)
                mask_los += mask_loss.item()
                label_los += label_loss.item()
                total_los += loss.item()
                experiment.log({
                    'mask loss': mask_loss.item(),
                    'label loss': label_loss.item(),
                    'train loss': loss.item(),
                    'step': global_step,
                    'epoch': epoch
                })
                pbar.set_postfix(
                    **{'total loss': loss.item(), 'mask loss': mask_loss.item(), 'label loss': label_loss.item()})

                # Evaluation round
                division_step = len(train_set) // (10 * batch_size)
                if division_step > 0 and global_step % division_step == 0:
                    pbar.set_postfix(**{'total_loss': total_los / division_step, 'mask_loss': mask_los / division_step,
                                        'label_loss': label_los / division_step})
                    total_los, mask_los, label_los = 0, 0, 0

                    histograms = {}
                    valid_score, valid_loss, map, ap, OP, OR, OF1, CP, CR, CF1, accuracy, precision, recall, f1 = evaluate(
                        model, valid_loader, device, amp)
                    # scheduler.step(valid_score)
                    logging.info(f'Validation Dice: {valid_score:.4f} \t Loss: {valid_loss:.4f} \t Acc: {accuracy:.3f}'
                                 f'\t SP: {precision:.3f} \t SR: {recall:.3f} \t SF1: {f1:.3f}\n'
                                 f'\t\tOP: {OP:.3f} \t OR: {OR:.3f} \t OF1: {OF1:.3f} \t CP: {CP:.3f} \t CR: {CR:.3f} \t CF1: {CF1:.3f}\n'
                                 f'\t\tmAP: {map:.2f} \t AP: {ap}')

                    experiment.log({
                        'learning rate': optimizer.param_groups[0]['lr'],
                        'validation Dice': valid_score,
                        'validation Loss': valid_loss,
                        'mAP': map,
                        'accuracy': accuracy,
                        'precision': precision,
                        'recall': recall,
                        'f1': f1,
                        'OP': OP,
                        'OR': OR,
                        'OF1': OF1,
                        'CP': CP,
                        'CR': CR,
                        'CF1': CF1,
                        'step': global_step,
                        'epoch': epoch,
                        **histograms
                    })

                    if save_checkpoint:
                        Path(dir_checkpoint).mkdir(parents=True, exist_ok=True)
                        state_dict = model.state_dict()
                        torch.save(state_dict, str(dir_checkpoint + f'epoch{epoch}_dice{valid_score:.4f}_p{precision:.4f}_f{f1:.4f}_r{recall:.4f}_a{accuracy:.4f}_map{map:.2f}.pth'))
                        logging.info(f'Checkpoint {epoch} saved!')
            # scheduler.step(epoch + 1)


def get_args():
    parser = argparse.ArgumentParser(description='Train the SSCNet on images and target masks')
    parser.add_argument('--epochs', '-e', metavar='E', type=int, default=100, help='Number of epochs')
    parser.add_argument('--batch-size', '-b', dest='batch_size', metavar='B', type=int, default=2, help='Batch size')
    parser.add_argument('--learning-rate', '-l', dest='lr', metavar='LR', type=float, default=1e-6, help='Learning rate')
    parser.add_argument('--load', '-f', type=str,  default=False, help='Load model from a .pth file')
    parser.add_argument('--scale', '-s', type=float, default=1.0, help='Downscaling factor of the images')
    parser.add_argument('--amp', action='store_true', default=True, help='Use mixed precision or not')
    parser.add_argument('--classes', '-c', type=int, default=[1, 7], help='Number of classes for segmentation and classification')

    return parser.parse_args()


if __name__ == '__main__':
    args = get_args()

    logging.basicConfig(level=logging.INFO, format='%(levelname)s: %(message)s')
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    logging.info(f'Using device {device}')

    model = SSCNet(classes=args.classes).to(device=device, memory_format=torch.channels_last)
    print(model)

    if args.load:
        state_dict = torch.load(args.load, map_location=device)
        model.load_state_dict(state_dict, strict=False)
        logging.info(f'Model loaded from {args.load}')

    train(
        model=model,
        epochs=args.epochs,
        batch_size=args.batch_size,
        learning_rate=args.lr,
        device=device,
        img_scale=args.scale,
        amp=args.amp
    )
