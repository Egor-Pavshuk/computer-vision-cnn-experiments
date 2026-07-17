import torch

class CBAMBlock(torch.nn.Module):
    def __init__(self, in_channels, reduction=16):
        super().__init__()

        self.max_pool = torch.nn.AdaptiveMaxPool2d(1)

        self.avg_pool = torch.nn.AdaptiveAvgPool2d(1)

        self.mlp = torch.nn.Sequential(
            torch.nn.Linear(in_channels, in_channels // reduction, bias=False),
            torch.nn.ReLU(),
            torch.nn.Linear(in_channels // reduction, in_channels, bias=False)
        )

        self.channelSigmoid = torch.nn.Sigmoid()

        self.spatial = torch.nn.Sequential(
            torch.nn.Conv2d(2, 1, kernel_size=7, padding=3, bias=False),
            torch.nn.Sigmoid()
        )

    def forward(self, x):
        bs, c, _, _ = x.shape

        # Channel Attention
        max_out = self.max_pool(x).view(bs, c)
        avg_out = self.avg_pool(x).view(bs, c)

        max_out_mlp = self.mlp(max_out)
        avg_out_mlp = self.mlp(avg_out)

        channel_attention = self.channelSigmoid(max_out_mlp + avg_out_mlp).view(bs, c, 1, 1)

        x = x * channel_attention

        # Spatial Attention
        avg_out = torch.mean(x, dim=1, keepdim=True)
        max_out, _ = torch.max(x, dim=1, keepdim=True)
        spatial = torch.cat([avg_out, max_out], dim=1)
        spatial_attention = self.spatial(spatial)

        x = x * spatial_attention

        return x