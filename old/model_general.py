import torch.nn as nn

from old.model_cross import CrossViT
from old.model_self import NonLocalBlock


class main_model(nn.Module):
    def __init__(self):
        self.self_attention = NonLocalBlock(128)
        self.cross_attention = CrossViT()

    def forward(self, x1, x2):
        x1_s = self.self_attention(x1)
        x2_s = self.self_attention(x2)
        x1_c, x2_c = self.cross_attention(x1, x2)




