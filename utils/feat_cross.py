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
        self.in_b = 64
        self.conv1 = ConvBlock(in_c=self.in_b, out_c=self.in_b//2, k=1)
        self.conv2 = nn.Conv2d(in_channels=self.in_b//2, out_channels=self.in_b, kernel_size=1)
        self.in_c = 12
        self.conv3 = ConvBlock(in_c=self.in_c, out_c = self.in_c//2, k=1)
        self.conv4 = nn.Conv2d(in_channels=self.in_c//2, out_channels=self.in_c, kernel_size=1)
        self.conv5 = nn.Conv2d(in_channels=23, out_channels=22, kernel_size=1)
        self.conv6 = nn.Conv2d(in_channels=22, out_channels=23, kernel_size=1)

    def get_attention(self, a):
        input_a = a
        a = a.unsqueeze(0)
        if a.size(1)==64:
            a = F.relu(self.conv1(a))
            a = self.conv2(a)
        elif a.size(1)==12:
            a = F.relu(self.conv3(a))
            a = self.conv4(a)
        a = a.squeeze(0)
        a = torch.mean(input_a*a, -1)
        a = F.softmax(a/0.05, dim=-1) + 1
        return a
    # f1, f2 [6, 2048, 18, 9]
    def forward(self, f1, f2): # if the channel
        b1, c, h, w = f1.size()
        b2 = f2.size(0)
        feat = torch.cat((f1, f2))
        feat = feat.view(b1+b2, c, -1)
        feat_norm = F.normalize(feat, p=2, dim=1, eps=1e-12)
        f1_norm = feat_norm[:b1]
        f2_norm = feat_norm[b1:]
        f1_norm = f1_norm.transpose(1, 2)
        a1 = torch.matmul(f1_norm, f2_norm)
        a2 = a1.transpose(1, 2)
        a = torch.cat((a1, a2))
        a = self.get_attention(a)
        feat = feat * a.unsqueeze(1)
        feat_1 = feat[:b1]
        feat_2 = feat[b1:]
        feat_1 = feat_1.view(b1, c, h, w)
        feat_2 = feat_2.view(b2, c, h, w)
        return feat_1, feat_2
        # b1, c1, h1, w1 = f1.size()
        # b2, c2, h2, w2 = f2.size()
        # f1 = f1.view(b1, c1, -1) # [32, 2048, 162]
        # f2 = f2.view(b2, c2, -1) # [32, 2048, 162]
        # f1_norm = F.normalize(f1, p=2, dim=1, eps=1e-12)
        # f2_norm = F.normalize(f2, p=2, dim=1, eps=1e-12)
        #
        # f1_norm = f1_norm.transpose(1, 2) # [32, 162, 2048] #[22, 2048, 162]
        # f2_norm = f2_norm # [32, 2048, 162] # [23, 2048, 162]
        # if f1_norm.size(0) != f2_norm.size(0):
        #     f2_norm = f2_norm.unsqueeze(0)
        #     f2_norm = self.conv5(f2_norm)
        #     f2_norm = f2_norm.squeeze(0)
        # a1 = torch.matmul(f1_norm, f2_norm)# [32, 162] # [22, 2048, 162] * [23, 2048, 162]  problem occoured
        # if a1.size(0) == 22:
        #     a2 = a1.unsqueeze(0)
        #     a2 = self.conv6(a2)
        #     a2 = a2.squeeze(0)
        #     a2 = a2.transpose(1, 2)
        # else:
        #     a2 = a1.transpose(1, 2)
        # a1 = self.get_attention(a1)
        # a2 = self.get_attention(a2)
        # f1 = f1 * a1.unsqueeze(1)
        # f1 = f1.view(b1, c1, h1, w1)
        # f2 = f2 * a2.unsqueeze(1)
        # f2 = f2.view(b2, c2, h2, w2)
        # return f1, f2



a = torch.randn(6, 2048, 18, 9)
b = torch.randn(6, 2048, 18, 9)
cross = feat_cross()
c, d = cross(a, b)




