from .resnet import ResNet
from .googlenet import GoogLeNet
from .vgg import VGG16

class ModelFactory():
    def __init__(self, name: str):
        name = name.lower()
        
        if name == "resnet_se":
            self.model = ResNet(attention="se")
        
        elif name == "resnet_cbam":
            self.model = ResNet(attention="cbam")
        
        elif name == "resnet":
            self.model = ResNet()
    
        elif name == "googlenet":
            self.model = GoogLeNet()
    
        elif name == "vgg":
            self.model = VGG16()

        else:
            raise ValueError(f"Unknown model: {name}")

    def get_model(self):
        return self.model