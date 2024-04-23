import logging
import numpy as np
import torch
from PIL import Image
from functools import lru_cache
from functools import partial
from itertools import repeat
from multiprocessing import Pool
from os import listdir
from os.path import splitext, isfile, join
from pathlib import Path
from torch.utils.data import Dataset
from tqdm import tqdm
import imgaug as ia
from imgaug import augmenters as iaa
from imgaug import parameters as iap
from torchvision import transforms


def load_image(filename):
    ext = splitext(filename)[1]
    if ext == '.npy':
        return Image.fromarray(np.load(filename))
    elif ext in ['.pt', '.pth']:
        return Image.fromarray(torch.load(filename).numpy())
    else:
        return Image.open(filename)


def unique_mask_values(idx, mask_dir, mask_suffix):
    mask_file = list(mask_dir.glob(idx + mask_suffix + '.*'))[0]
    mask = np.asarray(load_image(mask_file))
    if mask.ndim == 2:
        return np.unique(mask)
    elif mask.ndim == 3:
        mask = mask.reshape(-1, mask.shape[-1])
        return np.unique(mask, axis=0)
    else:
        raise ValueError(f'Loaded masks should have 2 or 3 dimensions, found {mask.ndim}')


def read_label(label_file, factor=0.01):
    with open(label_file, 'r') as file:
        line = file.readline().split()
        color = np.eye(3)[int(line[0])]
        colorthick = np.fromstring(' '.join(line[1:3]), sep=' ')
        crack = np.fromstring(' '.join(line[4:6]), sep=' ')
        label = np.concatenate((color, colorthick, crack))
        # label smoothing
        label_smooth = (1 - factor) * label
        label_smooth += sum(label) * factor / len(label)
        return label_smooth


def read_label_valid(label_file):
    with open(label_file, 'r') as file:
        line = file.readline().split()
        label = np.concatenate(
            (np.fromstring(' '.join(line[:3]), sep=' '), np.fromstring(' '.join(line[4:6]), sep=' ')))
        return label


class BasicDataset(Dataset):
    def __init__(self, images_dir: str, mask_dir: str, label_dir: str, scale: float = 1.0, mask_suffix: str = '',
                 valid: bool = False):
        self.images_dir = Path(images_dir)
        self.mask_dir = Path(mask_dir)
        self.label_dir = Path(label_dir)
        assert 0 < scale <= 1, 'Scale must be between 0 and 1'
        self.scale = scale
        self.mask_suffix = mask_suffix
        self.valid = valid
        self.ids = [splitext(file)[0] for file in listdir(images_dir) if
                    isfile(join(images_dir, file)) and not file.startswith('.')]
        if not self.ids:
            raise RuntimeError(f'No input file found in {images_dir}, make sure you put your images there')

        logging.info(f'Creating dataset with {len(self.ids)} examples')
        logging.info('Scanning mask files to determine unique values')
        with Pool() as p:
            unique = list(tqdm(
                p.imap(partial(unique_mask_values, mask_dir=self.mask_dir, mask_suffix=self.mask_suffix), self.ids),
                total=len(self.ids)
            ))

        self.mask_values = list(sorted(np.unique(np.concatenate(unique), axis=0).tolist()))
        logging.info(f'Unique mask values: {self.mask_values}')

    def __len__(self):
        return len(self.ids)

    @staticmethod
    def preprocess(img, mask, scale):
        if scale != 1.:
            w, h = img.size
            newW, newH = int(scale * w), int(scale * h)
            assert newW > 0 and newH > 0, 'Scale is too small, resized images would have no pixel'
            img = img.resize((newW, newH), resample=Image.BICUBIC)
            mask = mask.resize((newW, newH), resample=Image.NEAREST)

        img_nd = np.array(img)
        mask_nd = np.array(mask)
        seq = iaa.Sequential([
            iaa.Sometimes(0.75, iaa.Crop(px=(0, 10))),
            iaa.Sometimes(0.75, iaa.Affine(translate_percent={"x": (-0.1, 0.1), "y": (-0.1, 0.1)},
                                           rotate=(-10, 10),
                                           scale=(0.75, 1.2),
                                           shear={"x": (-5, 5), "y": (-5, 5)}
                                           )),
            iaa.Sometimes(0.75, iaa.Fliplr(0.5)),
            iaa.Sometimes(0.5, iaa.PiecewiseAffine(scale=(0.01, 0.025))),
            iaa.Sometimes(0.25, iaa.GaussianBlur((0, 0.5))),
            iaa.Sometimes(0.25, iaa.AdditiveGaussianNoise(loc=0, scale=(0.0, 0.05 * 255), per_channel=0.5))
        ])

        seg_map = ia.SegmentationMapsOnImage(mask_nd, shape=img_nd.shape)
        img_aug, seg_aug = seq(image=img_nd, segmentation_maps=seg_map)
        seg_map = seg_aug.get_arr()
        transform = transforms.ToTensor()
        img_trans = transform(img_aug)

        return img_trans, seg_map

    @staticmethod
    def preprocess_valid(img, mask, scale):
        if scale != 1.:
            w, h = img.size
            newW, newH = int(scale * w), int(scale * h)
            assert newW > 0 and newH > 0, 'Scale is too small, resized images would have no pixel'
            img = img.resize((newW, newH), resample=Image.BICUBIC)
            mask = mask.resize((newW, newH), resample=Image.NEAREST)

        img_nd = np.array(img)
        mask_nd = np.array(mask)
        transform = transforms.ToTensor()
        img_trans = transform(img_nd)
        return img_trans, mask_nd

    def __getitem__(self, idx):
        name = self.ids[idx]
        mask_file = list(self.mask_dir.glob(name + self.mask_suffix + '.*'))
        img_file = list(self.images_dir.glob(name + '.*'))
        label_file = list(self.label_dir.glob(name + '.*'))

        assert len(img_file) == 1, f'Either no image or multiple images found for the ID {name}: {img_file}'
        assert len(mask_file) == 1, f'Either no mask or multiple masks found for the ID {name}: {mask_file}'
        mask = load_image(mask_file[0])
        img = load_image(img_file[0])
        label = read_label(label_file[0])

        assert img.size == mask.size, f'Image and mask {name} should be the same size, but are {img.size} and {mask.size}'
        if self.valid:
            img, mask = self.preprocess_valid(img, mask, self.scale)
        else:
            img, mask = self.preprocess(img, mask, self.scale)

        return {
            'image': torch.as_tensor(img).float().contiguous(),
            'mask': torch.as_tensor(mask).long().contiguous(),
            'label': torch.as_tensor(label).float().contiguous()
        }
