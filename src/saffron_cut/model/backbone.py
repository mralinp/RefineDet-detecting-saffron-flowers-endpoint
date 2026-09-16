"""VGG16 'atrous' backbone, i.e. the same fully-convolutional reduced
VGGNet used by SSD and by the original RefineDet paper: conv1-conv5 from
ImageNet-pretrained VGG16, pool5 widened to stride-1 (3x3) so it stops
halving resolution, and the classifier's fc6/fc7 converted into dilated
convolutions and re-initialized from the pretrained fc6/fc7 weights via
kernel decimation (Liu et al., SSD, 2016).

Exposes four feature maps used as RefineDet's detection sources:
    conv4_3 (L2-normalized) -> stride  8
    fc7                     -> stride 16
    conv6_2                 -> stride 32
    conv7_2                 -> stride 64
"""

from __future__ import annotations

import torch
import torch.nn as nn
from torchvision.models import VGG16_Weights, vgg16


def _decimate(tensor: torch.Tensor, m: list[int | None]) -> torch.Tensor:
    """Subsample `tensor` along each dim by stride m[d] (keep every m[d]-th
    element); None leaves that dim untouched. Used to turn VGG's 4096-way
    fc6/fc7 weights into small dilated-conv kernels without throwing away
    the ImageNet-pretrained signal (see SSD, Liu et al. 2016, sec 3.1)."""
    for d, step in enumerate(m):
        if step is not None:
            idx = torch.arange(0, tensor.size(d), step)
            tensor = tensor.index_select(d, idx)
    return tensor


class L2Norm(nn.Module):
    """Per-channel L2 normalization with a learnable scale, applied to
    conv4_3 (its activations are much larger in magnitude than the deeper
    layers'). Same trick as the original SSD/RefineDet Caffe `normalize`
    layer."""

    def __init__(self, channels: int, init_scale: float = 10.0):
        super().__init__()
        self.weight = nn.Parameter(torch.full((channels,), float(init_scale)))
        self.eps = 1e-10

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        norm = x.pow(2).sum(dim=1, keepdim=True).sqrt() + self.eps
        return self.weight.view(1, -1, 1, 1) * (x / norm)


class VGGAtrousBackbone(nn.Module):
    out_channels = (512, 1024, 512, 256)  # conv4_3, fc7, conv6_2, conv7_2

    def __init__(self, pretrained: bool = True):
        super().__init__()
        weights = VGG16_Weights.IMAGENET1K_V1 if pretrained else None
        vgg = vgg16(weights=weights)
        features = list(vgg.features.children())

        # features[0:23]  -> conv1_1 .. relu4_3   (stride 8)
        # features[23:30] -> pool4, conv5_1..relu5_3 (stride 16 after pool4)
        self.conv1_4_3 = nn.Sequential(*features[0:23])
        self.conv5 = nn.Sequential(*features[23:30])
        # pool5: SSD/RefineDet widen this from 2x2/s2 to 3x3/s1 so it no
        # longer halves resolution (we already have enough downsampling).
        self.pool5 = nn.MaxPool2d(kernel_size=3, stride=1, padding=1)

        self.fc6 = nn.Conv2d(512, 1024, kernel_size=3, padding=6, dilation=6)
        self.fc7 = nn.Conv2d(1024, 1024, kernel_size=1)
        self.relu = nn.ReLU(inplace=True)

        if pretrained:
            self._init_fc_from_vgg_classifier(vgg)

        self.l2norm = L2Norm(512)

        self.extra6_1 = nn.Conv2d(1024, 256, kernel_size=1)
        self.extra6_2 = nn.Conv2d(256, 512, kernel_size=3, stride=2, padding=1)
        self.extra7_1 = nn.Conv2d(512, 128, kernel_size=1)
        self.extra7_2 = nn.Conv2d(128, 256, kernel_size=3, stride=2, padding=1)

        for m in [self.extra6_1, self.extra6_2, self.extra7_1, self.extra7_2, self.fc6, self.fc7]:
            if not (pretrained and m in (self.fc6, self.fc7)):
                nn.init.kaiming_normal_(m.weight, nonlinearity="relu")
                nn.init.zeros_(m.bias)

    def _init_fc_from_vgg_classifier(self, vgg: nn.Module) -> None:
        fc6_w = vgg.classifier[0].weight.data.view(4096, 512, 7, 7)
        fc6_b = vgg.classifier[0].bias.data
        fc7_w = vgg.classifier[3].weight.data.view(4096, 4096, 1, 1)
        fc7_b = vgg.classifier[3].bias.data

        self.fc6.weight.data = _decimate(fc6_w, [4, None, 3, 3])
        self.fc6.bias.data = _decimate(fc6_b, [4])
        self.fc7.weight.data = _decimate(fc7_w, [4, 4, None, None])
        self.fc7.bias.data = _decimate(fc7_b, [4])

    def forward(self, x: torch.Tensor) -> list[torch.Tensor]:
        c4_3 = self.conv1_4_3(x)
        c5_3 = self.conv5(c4_3)
        p5 = self.pool5(c5_3)
        fc7 = self.relu(self.fc7(self.relu(self.fc6(p5))))

        c6 = self.relu(self.extra6_2(self.relu(self.extra6_1(fc7))))
        c7 = self.relu(self.extra7_2(self.relu(self.extra7_1(c6))))

        return [self.l2norm(c4_3), fc7, c6, c7]
