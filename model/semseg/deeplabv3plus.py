import model.backbone.resnet as resnet

import torch
from torch import nn
import torch.nn.functional as F


class DeepLabV3Plus(nn.Module):
    def __init__(self):
        super(DeepLabV3Plus, self).__init__()

        self.backbone = resnet.resnet50(pretrained=True, replace_stride_with_dilation=[False, False, True])

        low_channels = 256
        high_channels = 2048

        self.head = ASPPModule(high_channels, [6, 12, 18])

        self.reduce = nn.Sequential(nn.Conv2d(low_channels, 48, 1, bias=False),
                                    nn.BatchNorm2d(48),
                                    nn.ReLU(True))

        self.fuse = nn.Sequential(nn.Conv2d(high_channels // 8 + 48, 256, 3, padding=1, bias=False),
                                  nn.BatchNorm2d(256),
                                  nn.ReLU(True),
                                  nn.Conv2d(256, 256, 3, padding=1, bias=False),
                                  nn.BatchNorm2d(256),
                                  nn.ReLU(True))

        self.classifier = nn.Conv2d(256, 2, 1, bias=True)
        self.binomial = torch.distributions.binomial.Binomial(probs=0.5)

    def forward(self, x1, x2, comp_drop=False):
        h, w = x1.shape[-2:]

        feats1 = self.backbone.base_forward(x1)
        c11, c14 = feats1[0], feats1[-1]
        
        feats2 = self.backbone.base_forward(x2)
        c21, c24 = feats2[0], feats2[-1]

        c1 = (c11 - c21).abs()
        c4 = (c14 - c24).abs()
        
        if comp_drop:
            # Complementary dropout for two strong augmentations (Unimatch v2).
            batch_pairs = c1.shape[0] // 2
            dim_c1 = c1.shape[1]  # 256
            dim_c4 = c4.shape[1]  # 2048
            
            # Generate dropout masks for c1
            dropout_mask1_c1 = self.binomial.sample((batch_pairs, dim_c1)).to(c1.device) * 2.0
            dropout_mask2_c1 = 2.0 - dropout_mask1_c1

            # Generate dropout masks for c4
            dropout_mask1_c4 = self.binomial.sample((batch_pairs, dim_c4)).to(c4.device) * 2.0
            dropout_mask2_c4 = 2.0 - dropout_mask1_c4

            # Half of the batch pairs are kept without dropout
            num_kept = int(batch_pairs * 0.5)
            kept_indexes = torch.randperm(batch_pairs, device=c1.device)[:num_kept]
            dropout_mask1_c1[kept_indexes, :] = 1.0
            dropout_mask2_c1[kept_indexes, :] = 1.0
            dropout_mask1_c4[kept_indexes, :] = 1.0
            dropout_mask2_c4[kept_indexes, :] = 1.0

            dropout_mask_c1 = torch.cat((dropout_mask1_c1, dropout_mask2_c1))
            dropout_mask_c4 = torch.cat((dropout_mask1_c4, dropout_mask2_c4))

            c1 = c1 * dropout_mask_c1[..., None, None]
            c4 = c4 * dropout_mask_c4[..., None, None]

            out = self._decode(c1, c4)
            out = F.interpolate(out, size=(h, w), mode="bilinear", align_corners=True)

            return out

        out = self._decode(c1, c4)
        out = F.interpolate(out, size=(h, w), mode="bilinear", align_corners=True)

        return out

    def _decode(self, c1, c4):
        c4 = self.head(c4)
        c4 = F.interpolate(c4, size=c1.shape[-2:], mode="bilinear", align_corners=True)

        c1 = self.reduce(c1)

        feature = torch.cat([c1, c4], dim=1)
        feature = self.fuse(feature)

        out = self.classifier(feature)

        return out


def ASPPConv(in_channels, out_channels, atrous_rate):
    block = nn.Sequential(nn.Conv2d(in_channels, out_channels, 3, padding=atrous_rate,
                                    dilation=atrous_rate, bias=False),
                          nn.BatchNorm2d(out_channels),
                          nn.ReLU(True))
    return block


class ASPPPooling(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(ASPPPooling, self).__init__()
        self.gap = nn.Sequential(nn.AdaptiveAvgPool2d(1),
                                 nn.Conv2d(in_channels, out_channels, 1, bias=False),
                                 nn.BatchNorm2d(out_channels),
                                 nn.ReLU(True))

    def forward(self, x):
        h, w = x.shape[-2:]
        pool = self.gap(x)
        return F.interpolate(pool, (h, w), mode="bilinear", align_corners=True)


class ASPPModule(nn.Module):
    def __init__(self, in_channels, atrous_rates):
        super(ASPPModule, self).__init__()
        out_channels = in_channels // 8
        rate1, rate2, rate3 = atrous_rates

        self.b0 = nn.Sequential(nn.Conv2d(in_channels, out_channels, 1, bias=False),
                                nn.BatchNorm2d(out_channels),
                                nn.ReLU(True))
        self.b1 = ASPPConv(in_channels, out_channels, rate1)
        self.b2 = ASPPConv(in_channels, out_channels, rate2)
        self.b3 = ASPPConv(in_channels, out_channels, rate3)
        self.b4 = ASPPPooling(in_channels, out_channels)

        self.project = nn.Sequential(nn.Conv2d(5 * out_channels, out_channels, 1, bias=False),
                                     nn.BatchNorm2d(out_channels),
                                     nn.ReLU(True))

    def forward(self, x):
        feat0 = self.b0(x)
        feat1 = self.b1(x)
        feat2 = self.b2(x)
        feat3 = self.b3(x)
        feat4 = self.b4(x)
        y = torch.cat((feat0, feat1, feat2, feat3, feat4), 1)
        return self.project(y)
