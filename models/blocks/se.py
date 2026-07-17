import torch

class SEBlock(torch.nn.Module):
    def __init__(self, in_channels, reduction=16):
        super().__init__()

        self.squeeze = torch.nn.AdaptiveAvgPool2d(1)

        self.exitation = torch.nn.Sequential(
            torch.nn.Linear(in_channels, in_channels // reduction, bias=False),
            torch.nn.ReLU(),
            torch.nn.Linear(in_channels // reduction, in_channels, bias=False),
            torch.nn.Sigmoid()
        )

    def forward(self, x):
        bs, c, _, _ = x.size()
        y = self.squeeze(x).view(bs, c)
        y = self.exitation(y).view(bs, c, 1, 1)
        x = x * y.expand_as(x)
        return x