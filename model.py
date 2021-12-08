import torch
import torch.nn as nn
from torch.nn import init

from utils.feat_cross import feat_cross
from utils.feat_self import feat_self
from utils.loss import *
from utils.resnet import resnet50

def weights_init_kaiming(m):
    classname = m.__class__.__name__
    # print(classname)
    if classname.find('Conv') != -1:
        init.kaiming_normal_(m.weight.data, a=0, mode="fan_in")
    elif classname.find('Linear') != -1:
        init.kaiming_normal_(m.weight.data, a=0, mode='fan_out')
        init.zeros_(m.bias.data)
    elif classname.find('BatchNorm1d') != -1:
        init.normal_(m.weight.data, 1.0, 0.01)
        init.zeros_(m.bias.data)

def weights_init_classifier(m):
    classname = m.__class__.__name__
    if classname.find('Linear') != -1:
        init.normal_(m.weight.data, 0, 0.001)
        if m.bias:
            init.zeros_(m.bias.data)

class visible_module(nn.Module):
    def __init__(self, arch='resnet50'):
        super(visible_module, self).__init__()

        model_v = resnet50(pretrained=True,
                           last_conv_stride=1, last_conv_dilation=1)
        # avg pooling to global pooling
        self.visible = model_v

    def forward(self, x):
        x = self.visible.conv1(x)
        x = self.visible.bn1(x)
        x = self.visible.relu(x)
        x = self.visible.maxpool(x)
        return x


class thermal_module(nn.Module):
    def __init__(self, arch='resnet50'):
        super(thermal_module, self).__init__()

        model_t = resnet50(pretrained=True,
                           last_conv_stride=1, last_conv_dilation=1)
        # avg pooling to global pooling
        self.thermal = model_t

    def forward(self, x):
        x = self.thermal.conv1(x)
        x = self.thermal.bn1(x)
        x = self.thermal.relu(x)
        x = self.thermal.maxpool(x)
        return x


class base_resnet(nn.Module):
    def __init__(self, arch='resnet50'):
        super(base_resnet, self).__init__()

        model_base = resnet50(pretrained=True,
                              last_conv_stride=1, last_conv_dilation=1)
        # avg pooling to global pooling
        model_base.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        self.base = model_base

    def forward(self, x):
        x = self.base.layer1(x)
        x = self.base.layer2(x)
        x = self.base.layer3(x)
        x = self.base.layer4(x)
        return x

class network(nn.Module):
    def __init__(self,class_num, args, arch='resnet50'):
        super(network, self).__init__()
        self.thermal_module = thermal_module(arch=arch)
        self.visible_module = visible_module(arch=arch)
        self.base_resnet = base_resnet(arch=arch)

        self.self_attention = feat_self()
        self.cross_attention = feat_cross()

        pool_dim = 2048
        self.bottleneck = nn.BatchNorm1d(pool_dim)
        self.bottleneck.bias.requires_grad_(False)

        self.classifier = nn.Linear(pool_dim, class_num, bias=False)
        self.bottleneck.apply(weights_init_kaiming)
        self.classifier.apply(weights_init_classifier)
        self.pool = GeMP()


    def forward(self, x1, x2, mode=0):
        if mode == 0:
            x1 = self.visible_module(x1)
            x2 = self.thermal_module(x2)
            x = torch.cat((x1, x2), dim=0)
        elif mode == 1:
            x = self.visible_module(x1)
        elif mode == 2:
            x = self.thermal_module(x2)

        # 这里的x应该是包含了还几张图片的可见光和红外光特征 batch_size * channel * height * width，需要做的就是将两个模态的特征分开，然后进行注意力特征提取
        x = self.base_resnet(x)
        x_p = self.pool(x)
        batch_size, fdim, h, w = x.shape
        xv = x[:batch_size//2]
        xt = x[batch_size//2:]
        xv_s = self.self_attention(xv)
        xt_s = self.self_attention(xt)
        xv_c, xt_c = self.cross_attention(xv, xt)

        # consider how to fusion the feature can get better performance
        xv_recon = xv_s + 2 * xv_c
        xt_recon = xt_s + 2 * xt_c
        feat_recon = torch.cat([xv_recon, xt_recon], dim=0)
        recon_p = self.pool(feat_recon) # lack of pooling operation
        clsid_recon = self.classifier(self.bottleneck(recon_p)) # return it
        clsid = self.classifier(self.bottleneck(x_p)) # return it

        # loss = id loss + id_con loss + triplet loss
        '''
        then we need to return feats as following:
        feat_pooling_v, feat_pooling_t
        refeat_pooling_v, refeat_pooling_t
        or we can return other essential parameters 
        '''
        return {
            'cls_id': clsid,
            'cls_id_recon': clsid_recon,
            'x_p': x_p,
            'recon_p': recon_p
        }


if __name__ == '__main__':
    net = network(class_num=20)
    a = torch.randn(8,3,255,255)
    b = torch.randn(8,3,255,255)
    out = net(a, b)
    print(out.shape)

