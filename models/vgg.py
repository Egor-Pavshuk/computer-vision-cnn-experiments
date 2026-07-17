import torch.nn as nn
from .blocks import Conv2dBlock

class VGG16(nn.Module):
    def __init__(self, attention = None):
        super().__init__()

        self.features = nn.Sequential(

            Conv2dBlock(3, 64, kernel_size=3, stride=1, padding=1),
            Conv2dBlock(64, 64, kernel_size=3, stride=1, padding=1),

            nn.MaxPool2d(kernel_size=2, stride=2),

            Conv2dBlock(64, 128, kernel_size=3, stride=1, padding=1),
            Conv2dBlock(128, 128, kernel_size=3, stride=1, padding=1),

            nn.MaxPool2d(kernel_size=2, stride=2),

            Conv2dBlock(128, 256, kernel_size=3, stride=1, padding=1),
            Conv2dBlock(256, 256, kernel_size=3, stride=1, padding=1),
            Conv2dBlock(256, 256, kernel_size=3, stride=1, padding=1),

            nn.MaxPool2d(kernel_size=2, stride=2),

            Conv2dBlock(256, 512, kernel_size=3, stride=1, padding=1),
            Conv2dBlock(512, 512, kernel_size=3, stride=1, padding=1),
            Conv2dBlock(512, 512, kernel_size=3, stride=1, padding=1),
        )

        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Linear(4*4*512, 512),
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(512, 256),
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(256, 10),
        )

    def forward(self, x):
        x = self.features(x)
        x = self.classifier(x)
        return x

