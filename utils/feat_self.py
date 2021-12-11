import torch
from torch import nn

'''
    @Description:
        feat-self is follewed by Non-Local Block which is used in getting the self-attention of an image.
        https://arxiv.org/abs/1711.07971
'''


class feat_self(nn.Module):
    def __init__(self):
        super(feat_self, self).__init__()
        in_channels = 2048
        reduction_ratio = 2
        self.in_channels = in_channels
        self.inter_channels = in_channels // reduction_ratio
        self.g = nn.Sequential(
            nn.Conv2d(in_channels = self.in_channels, out_channels= self.inter_channels, kernel_size=1, stride=1, padding=0)
        )
        self.W = nn.Sequential(
            nn.Conv2d(in_channels = self.inter_channels, out_channels=self.in_channels, kernel_size=1, stride=1, padding=0),
            nn.BatchNorm2d(self.in_channels)
        )
        nn.init.constant_(self.W[1].weight, 0.0)
        nn.init.constant_(self.W[1].bias, 0.0)

        self.theta = nn.Conv2d(in_channels=self.in_channels, out_channels=self.inter_channels, kernel_size=1, stride=1, padding=0)

        self.phi = nn.Conv2d(in_channels=self.in_channels, out_channels=self.inter_channels, kernel_size=1, stride=1, padding=0)

    def forward(self, x):
        batch_size = x.size(0)
        g_x = self.g(x).view(batch_size, self.inter_channels, -1)
        g_x = g_x.permute(0, 2, 1)

        theta_x = self.theta(x).view(batch_size, self.inter_channels, -1)
        theta_x = theta_x.permute(0, 2, 1)
        phi_x = self.phi(x).view(batch_size, self.inter_channels, -1)
        f = torch.matmul(theta_x, phi_x)
        N = f.size(-1)
        f_div_C = torch.nn.functional.softmax(f, dim=-1)
        # f_div_C = f / N # softmax的替代品

        y = torch.matmul(f_div_C, g_x)
        y = y.permute(0, 2, 1).contiguous()
        y = y.view(batch_size, self.inter_channels, *x.size()[2:])
        W_y = self.W(y)
        z = W_y + x
        return z

a = torch.randn(32, 2048, 18, 9)
self = feat_self()
a = self(a)
