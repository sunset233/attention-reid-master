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
# class feat_cross(nn.Module):
#     def __init__(self):
#         super(feat_cross, self).__init__()
#         self.in_channels = 2048
#         stride = 3
#         self.downsample = nn.Sequential(
#             nn.Conv2d(self.in_channels, self.in_channels, kernel_size=3, stride=stride),
#             nn.AdaptiveAvgPool2d((3, 3)),
#             nn.BatchNorm2d(self.in_channels))
#
#     def xcorr_depthwise(self, x, kernel):
#         """depthwise cross correlation
#         """
#         channel = kernel.size(0)
#         x = torch.mul(x, kernel)
#         kernel = self.downsample(kernel)
#         x = x.view(-1, 1, x.size(1), x.size(2))
#         kernel = kernel.view(-1, 1, kernel.size(1), kernel.size(2))
#         out = F.conv2d(x, kernel, groups=1 *channel)
#         out = out.view(channel, out.size(2), out.size(3))
#         return out
#         # # x.shape [2048, 18, 9]  kernel.shape [2048, 18, 9]
#         # x = x.view(1, batch * channel, x.size(2), x.size(3))
#         # kernel = kernel.view(batch * channel, 1, kernel.size(2), kernel.size(3))
#         # out = F.conv2d(x, kernel, groups=batch * channel)
#         # out = out.view(batch, channel, out.size(2), out.size(3))
#         # return out
#
#     '''
#     @Description:
#         Cross-attention is modified by depth-wise correlation matrix to extract correlation map
#     '''
#     # f1.shape [32, 2048, 18, 9]  f2.shape [32, 2048, 18, 9]
#     def forward(self, f1, f2):
#         f1_origin = f1
#         f2_origin = f2
#         batch = f2.size(0)
#         channel = f2.size(1)
#         f1 = f1.view(1, -1, f1.size(2), f1.size(3))
#         f2 = self.downsample(f2)
#         # before downsample [32, 2048, 18, 9] after downsample [32, 2048, 3, 3]
#         f2 = f2.view(-1, 1, f2.size(2), f2.size(3))
#         features = F.conv2d(f1, f2, stride=1, padding=1, groups=batch*channel)
#         # f1.shape [1, 65536, 18, 9] f2.shape [65536, 1, 3, 3] features.shape [1, 65536, 13, 7]
#         # features = f1 + f2
#         batch_size, c, h, w = features.size(0), features.size(1), features.size(2), features.size(3)
#         features -= features.min(dim=-1, keepdim=True)[0]
#         features /= features.max(dim=-1, keepdim=True)[0] + 1e-12
#         mask = features.view(batch, channel, h, w)
#         # f1_origin.shape [32, 2048, 18, 9]  f2_origin.shape [32, 2048, 18, 9] mask.shape [32, 2048, 18, 9]
#         f1_cross = f1_origin * mask
#         f2_cross = f2_origin * mask
#         return f1_cross, f2_cross

class ConvBlock(nn.Module):
    def __init__(self, in_c, out_c, k, s=1, p=0):
        super(ConvBlock, self).__init__()
        self.conv = nn.Conv2d(in_c, out_c, k, padding=p)
        self.bn = nn.BatchNorm2d(out_c)

    def forward(self, x):
       return self.bn(self.conv(x))

class feat_cross(nn.Module):
    def __init__(self):
        super(feat_cross, self).__init__()
        self.in_b = 32
        self.conv1 = ConvBlock(in_c=self.in_b, out_c=self.in_b//2, k=1)
        self.conv2 = nn.Conv2d(in_channels=self.in_b//2, out_channels=self.in_b, kernel_size=1)
        self.in_c = 6
        self.conv3 = ConvBlock(in_c=self.in_c, out_c = self.in_c//2, k=1)
        self.conv4 = nn.Conv2d(in_channels=self.in_c//2, out_channels=self.in_c, kernel_size=1)

    def get_attention(self, a):
        input_a = a
        a = a.unsqueeze(0)
        if a.size(1)==32:
            a = F.relu(self.conv1(a))
            a = self.conv2(a)
        elif a.size(1)==6:
            a = F.relu(self.conv3(a))
            a = self.conv4(a)
        a = a.squeeze(0)
        a = torch.mean(input_a*a, -1)
        a = F.softmax(a/0.05, dim=-1) + 1
        return a
    # f1, f2 [6, 2048, 18, 9]
    def forward(self, f1, f2):
        b1, c1, h1, w1 = f1.size()
        b2, c2, h2, w2 = f2.size()
        f1 = f1.view(b1, c1, -1) # [32, 2048, 162]
        f2 = f2.view(b2, c2, -1) # [32, 2048, 162]
        f1_norm = F.normalize(f1, p=2, dim=1, eps=1e-12)
        f2_norm = F.normalize(f2, p=2, dim=1, eps=1e-12)

        f1_norm = f1_norm.transpose(1, 2) # [32, 162, 2048] #[22, 2048, 162]
        f2_norm = f2_norm # [32, 2048, 162] # [23, 2048, 162]

        a1 = torch.matmul(f1_norm, f2_norm) # [32, 162] # [22, 2048, 162] * [23, 2048, 162]  problem occoured
        a2 = a1.transpose(1, 2)

        a1 = self.get_attention(a1) # [32, 162]
        a2 = self.get_attention(a2) # [32, 162]
        f1 = f1 * a1.unsqueeze(1)
        f1 = f1.view(b1, c1, h1, w1)
        f2 = f2 * a2.unsqueeze(1)
        f2 = f2.view(b2, c2, h2, w2)
        return f1, f2



a = torch.randn(6, 2048, 18, 9)
b = torch.randn(6, 2048, 18, 9)
cross = feat_cross()
c, d = cross(a, b)




