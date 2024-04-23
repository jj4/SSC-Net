import torch
import torch.nn as nn


class DFL(nn.Module):
    def __init__(self, k=8, nclass=7):
        super().__init__()
        # k channels for one class, nclass is total classes, therefore k * nclass for conv6
        self.k = k
        self.nclass = nclass

        # G-Stream
        self.cls5 = nn.Sequential(
            nn.Conv2d(2048, self.nclass, kernel_size=1, stride=1, padding=0),
            nn.BatchNorm2d(num_features=self.nclass),
            nn.ReLU(inplace=True),
            nn.AdaptiveAvgPool2d((1, 1)),
        )

        # P-Stream
        self.conv6 = nn.Conv2d(1024, self.k * self.nclass, kernel_size=1, stride=1, padding=0)
        self.pool6 = nn.AdaptiveMaxPool2d((1, 1))
        # self.pool6 = nn.MaxPool2d((24, 18), stride=(24, 18))  # , return_indices=True)
        self.cls6 = nn.Sequential(
            nn.Conv2d(self.k * self.nclass, self.nclass, kernel_size=1, stride=1, padding=0),
            nn.AdaptiveAvgPool2d((1, 1)),
        )

        # Side-branch
        self.ccpool = nn.AvgPool1d(kernel_size=self.k, stride=self.k, padding=0)

    def forward(self, *features):
        # Stem: Feature extraction
        conv5 = features[-1]  # torch.Size([10, 2048, 24, 18])
        conv4 = features[-2]  # torch.Size([10, 1024, 48, 36])

        # G-stream
        g = self.cls5(conv5)  # torch.Size([10, 10, 1, 1])
        g = torch.flatten(g, 1)  # torch.Size([10, 10])

        # P-stream
        p = self.conv6(conv4)  # torch.Size([10, 100, 48, 36])
        p = self.pool6(p)  # torch.Size([10, 100, 1, 1])
        s = p
        p = self.cls6(p)  # torch.Size([10, 10, 1, 1])
        p = torch.flatten(p, 1)  # torch.Size([10, 10])

        # Side-branch
        s = torch.flatten(s, 2).transpose(1, 2)  # torch.Size([10, 1, 100])
        s = self.ccpool(s)  # torch.Size([10, 1, 10])
        s = torch.flatten(s, 1)  # torch.Size([10, 10])
        return g, p, s
