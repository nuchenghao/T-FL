from torch import nn
from torch.nn import functional as F


class MLP(nn.Module):
    def __init__(self):
        super().__init__()
        self.hidden = nn.Linear(10, 5)
        self.output = nn.Linear(5, 2)

    def forward(self, x):
        return self.output(F.relu(self.hidden(x)))


net = MLP()


def init_weights(m):
    """初始化网络的层"""
    if type(m) == nn.Linear or type(m) == nn.Conv2d:
        nn.init.xavier_uniform_(m.weight)
