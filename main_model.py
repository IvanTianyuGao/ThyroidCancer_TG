import torch
from torch import nn
from torchvision.models import resnet50, densenet121, inception_v3

#CNN model preparation and pre-trained parameter

class ResNet50(nn.Module):
    def __init__(self):
        super().__init__()
        base_model = resnet50(pretrained=False)
        layers = list(base_model.children())
        self.resnet50 = nn.Sequential(*layers[:9])

    def forward(self, x):
        return self.resnet50(x)
