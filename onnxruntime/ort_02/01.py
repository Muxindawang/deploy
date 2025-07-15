import torch
import torch.nn as nn
import torchvision.transforms as T
from PyQt5.QtGui.QRawFont import weight
from holoviews.ipython.widgets import progress
from torchvision.models import resnet18, ResNet18_Weights

class Predictor(nn.Module):
    def __init__(self):
        super().__init__()
        weights = ResNet18_Weights.DEFAULT
        self.resnet18 = resnet18(weights=weights, progress=False).eval()

        self.transforms = weights.transforms()

    def forward(self, x: torch.tensor):
        with torch.no_grad():
            x = self.transforms(x)
            y_pred = self.resnet18(x)
            return y_pred.argmax(dim=1)


