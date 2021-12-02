import torch
import torch.nn as nn
import math
import torch.nn.functional as F
'''
    @Description:
        feat_cross is followed by CAM which is used in few-shot classification
        https://arxiv.org/abs/1910.07677v1
        It can get the cross-attention between two images tensors.
'''
class feat_cross(nn.Module):
    def __init__(self):
        self.in_channels = 2048
        stride = 3
        super(feat_cross, self).__init__()
        self.downsample = nn.Sequential(
            nn.Conv2d(self.in_channels, self.in_channels/2, kernel_size=3, stride=stride),
            nn.BatchNorm2d(self.in_channels/2))

    def xcorr_depthwise(self, x, kernel):
        """depthwise cross correlation
        """
        channel = kernel.size(0)
        x = torch.mul(x, kernel)
        kernel = self.downsample(kernel)
        x = x.view(-1, 1, x.size(1), x.size(2))
        kernel = kernel.view(-1, 1, kernel.size(1), kernel.size(2))
        out = F.conv2d(x, kernel, groups=1 *channel)
        out = out.view(channel, out.size(2), out.size(3))
        return out
        # # x.shape [2048, 18, 9]  kernel.shape [2048, 18, 9]
        # x = x.view(1, batch * channel, x.size(2), x.size(3))
        # kernel = kernel.view(batch * channel, 1, kernel.size(2), kernel.size(3))
        # out = F.conv2d(x, kernel, groups=batch * channel)
        # out = out.view(batch, channel, out.size(2), out.size(3))
        # return out

    '''
    @Description:
        Cross-attention is modified by depth-wise correlation matrix to extract correlation map
    '''

    def forward(self, f1, f2):
        batch = f2.size(0)
        channel = f2.size(1)
        f1 = f1.view(1, batch*channel, f1.size(2), f1.size(3))
        f2 = self.downsample(f2)
        f2 = f2.view(batch*channel, 1, f2.size(2), f2.size(3))
        features = F.conv2d(f1, f2, groups=batch*channel)
        # features = f1 * f2
        batch_size, c, h, w = features.size(0), features.size(1), features.size(2), features.size(3)
        features -= features.min(dim=-1, keepdim=True)[0]
        features /= features.max(dim=-1, keepdim=True)[0] + 1e-12
        mask = features.view(batch_size, c, h, w)
        f1_cross = torch.mul(f1, mask)
        f2_cross = torch.mul(f2, mask)
        return f1_cross, f2_cross



