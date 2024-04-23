import argparse
import logging
import os
import torch
from torch.utils.data import DataLoader
from evaluate import evaluate
from model import SSCNet
from utils import BasicDataset
import warnings

warnings.filterwarnings('ignore')

dir_img = './data/imgs/'
dir_mask = './data/masks/'
dir_label = './data/labels/'
dir_checkpoint = './checkpoints/'


def test(model, device, batch_size: int = 1, img_scale: float = 1.0, amp: bool = True):
    # 1. Create dataset
    test_set = BasicDataset(dir_img + 'valid', dir_mask + 'valid', dir_label + 'valid', img_scale, valid=True)
    # 2. Create data loaders
    loader_args = dict(batch_size=batch_size, num_workers=os.cpu_count(), pin_memory=True)
    test_loader = DataLoader(test_set, shuffle=False, drop_last=True, **loader_args)
    # 4. Begin testing
    dice, map, ap, accuracy, precision, recall, f1, t, p = evaluate(model, test_loader, device, amp)
    return dice, map, ap, accuracy, precision, recall, f1, t, p


def get_args():
    parser = argparse.ArgumentParser(description='Train the SSCNet on images and target masks')
    parser.add_argument('--batch-size', '-b', dest='batch_size', metavar='B', type=int, default=1, help='Batch size')
    parser.add_argument('--load', '-f', type=str, default=False, help='Load model from a .pth file')
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

    if args.load:
        state_dict = torch.load(args.load, map_location=device)
        model.load_state_dict(state_dict)
        logging.info(f'Model loaded from {args.load}')

    dice, map, ap, accuracy, precision, recall, f1, t, p = test(model=model, batch_size=args.batch_size, device=device, img_scale=args.scale, amp=args.amp)
    logging.info('Validation Dice: {} \t Acc.: {} \t Prec.: {} \t Rec.: {} \t F1: {} \n\t\t mAP: {} \t AP: {}'.
                 format(round(dice.item(), 4), round(accuracy.item(), 3), round(precision.item(), 3),
                        round(recall.item(), 3), round(f1.item(), 3), round(map.item(), 2), ap))
    print('t', t, 'p', p)
