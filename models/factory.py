from .resnet import ResNet
from .googlenet import GoogLeNet
from .vgg import VGG16


def create_model(name: str):
    name = name.lower()

    if name == "resnet_se":
        return ResNet(attention="se")
    
    if name == "resnet_cbam":
        return ResNet(attention="cbam")
    
    if name == "resnet":
        return ResNet()

    if name == "googlenet":
        return GoogLeNet()

    if name == "vgg":
        return VGG16()

    raise ValueError(f"Unknown model: {name}")