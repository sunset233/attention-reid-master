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
        super(feat_cross, self).__init__()


    def xcorr_depthwise(self, x, kernel):
        """depthwise cross correlation
        """
        batch = kernel.size(0)
        channel = kernel.size(1)
        x = torch.mul(x, kernel)
        # x.shape [2048, 18, 9]  kernel.shape [2048, 18, 9]
        x = x.view(1, batch * channel, x.size(2), x.size(3))
        kernel = kernel.view(batch * channel, 1, kernel.size(2), kernel.size(3))
        out = F.conv2d(x, kernel, groups=batch * channel)
        out = out.view(batch, channel, out.size(2), out.size(3))
        return out


    def forward(self, f1, f2):

        features = f1 * f2
        batch_size, c, h, w = features.size(0), features.size(1), features.size(2), features.size(3)
        features -= features.min(dim=-1, keepdim=True)[0]
        features /= features.max(dim=-1, keepdim=True)[0] + 1e-12
        mask = features.view(batch_size, c, h, w)
        f1_cross = torch.mul(f1, mask)
        f2_cross = torch.mul(f2, mask)
        return f1_cross, f2_cross



