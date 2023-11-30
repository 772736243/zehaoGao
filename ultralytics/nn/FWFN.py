import torch
from torch import nn
from ..nn.modules import Conv

class Fusion(nn.Module):
    def __init__(self, inc_list, fusion='weight') -> None:
        super().__init__()

        assert fusion in ['weight']
        self.fusion = fusion
        self.fusion_conv = nn.ModuleList([Conv(inc, inc, 1) for inc in inc_list])

    def forward(self, x):
        for i in range(len(x)):
            x[i] = self.fusion_conv[i](x[i])
        if self.fusion == 'weight':
            return torch.sum(torch.stack(x, dim=0), dim=0)