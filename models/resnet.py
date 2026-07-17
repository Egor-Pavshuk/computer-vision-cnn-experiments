import torch.nn as nn
from .blocks import ResidualBlock

class ResNet(nn.Module):
    def __init__(self, attention = None):
        super().__init__()

        self.features = nn.Sequential(

            nn.Conv2d(3, 64, kernel_size=7, stride=2, padding=3),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=3, stride=2, padding=1),

            ResidualBlock(64, 64, kernel_size=3, stride=1, padding=1, attention=attention),
            ResidualBlock(64, 64, kernel_size=3, stride=1, padding=1, attention=attention),

            ResidualBlock(64, 128, kernel_size=3, stride=2, padding=1, attention=attention),
            ResidualBlock(128, 128, kernel_size=3, stride=1, padding=1, attention=attention),

            nn.AdaptiveAvgPool2d((4, 4)),
        )

        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Linear(4*4*128, 10),
        )

    def forward(self, x):
        x = self.features(x)
        x = self.classifier(x)
        return x

