import torch
import torch.nn as nn
import torch.nn.functional as F
import segmentation_models_pytorch as smp
from .header import MHA, DFL
from utils.utils import get_bounding_box, to_varabile
from roi_align.roi_align import RoIAlign


class SSCNet(nn.Module):
    def __init__(self, classes):
        super().__init__()
        self.seg_classes = classes[0]
        self.cls_classes = classes[1]

        backbone = smp.Unet(
            encoder_name='se_resnext50_32x4d',
            encoder_weights='imagenet',
            classes=self.seg_classes,
        )
        self.encoder = backbone.encoder
        self.decoder = backbone.decoder
        self.segmentation_head = backbone.segmentation_head
        # self.dfl = DFL()
        # self.classification_head = ClassificationHead(in_channels=self.encoder.out_channels[-1],
        #                                               classes=self.cls_classes)
        self.classification_head = MHA(num_heads=6,
                                       lam=0.1,
                                       input_dim=self.encoder.out_channels[-1], num_classes=self.cls_classes)

    def forward(self, x):
        features = self.encoder(x)
        out = self.decoder(*features)
        mask = self.segmentation_head(out)
        #
        #         ╭───────────────────────── (not computing graident) ─────────────────────────╮
        #         ↓                                                                            ↑
        # input —⨂-> [encoder] -> features -> [decoder] -> outputs -> [segmentation head] -> masks
        #                 │
        #                 ╰--> masked features --┬-> [classification head] -> labels
        #                                        │
        #                                        ╰-> [detection head] -> boxes
        #
        mask_ = F.sigmoid(mask.detach()) >= 0.5
        mask_x = x.detach() * mask_
        boxes = get_bounding_box(mask_.detach() >= 0.5)
        roi = RoIAlign(512, 384)(x.contiguous().detach() * mask_.detach(), boxes, to_varabile(torch.arange(boxes.shape[0], dtype=torch.int32)))
        rois = self.encoder(roi)
        last_feature = self.fpn(*rois)
        # g, p, s = self.dfl(*rois)
        label = self.classification_head(last_feature)

        return {
            'mask': mask,
            'label': label
            # 'label': [g, p, s]
        }


class ClassificationHead(nn.Sequential):
    def __init__(self, in_channels, classes, dropout=0.5):
        pool = nn.AdaptiveAvgPool2d(1)
        flatten = nn.Flatten()
        dropout = nn.Dropout(p=dropout)
        linear = nn.Linear(in_channels, classes)
        super().__init__(pool, flatten, dropout, linear)


class FPNDecoder(nn.Module):
    def __init__(self, encoder_channels):
        super().__init__()
        self.p3 = FPNBlock(encoder_channels[2])
        self.p4 = FPNBlock(encoder_channels[3])
        self.p5 = FPNBlock(encoder_channels[4])

    def forward(self, *features):
        c2, c3, c4, c5 = features[-4:]
        p2 = c2
        p3 = self.p3(p2, c3)  # 512
        p4 = self.p4(p3, c4)  # 1024
        p5 = self.p5(p4, c5)  # 2048
        return p5


class FPNBlock(nn.Module):
    def __init__(self, in_channels):
        super().__init__()
        self.conv3 = nn.Sequential(
            nn.Conv2d(in_channels, in_channels * 2, kernel_size=3, stride=2, padding=1),
            nn.ReLU(inplace=True)
        )

    def forward(self, x, skip=None):
        x = self.conv3(x)
        x = x + skip
        return x
