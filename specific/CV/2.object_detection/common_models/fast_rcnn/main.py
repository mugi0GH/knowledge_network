from torch import nn
from torchvision.models import vgg16,VGG16_Weights
from torchvision.models.detection.faster_rcnn import RegionProposalNetwork # ,AnchorGenerator

class backbone(nn.Module):
    def __init__(self, init_weights=True):
        super(backbone, self).__init__()
        # Load the pre-trained VGG model
        self.vgg = vgg16(weights=VGG16_Weights.DEFAULT)
        self.RPN = VGG16_Weights()
        # remove output layers and 
        del self.vgg.classifier,self.vgg.avgpool
        # Initialize weights if required
        if init_weights:
            self._initialize_weights()

    def forward(self, x):
        
        return self.vgg.features(x)
    
    def _initialize_weights(self):
        # Initialize or modify weights here if necessary
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.Linear):
                nn.init.normal_(m.weight, 0, 0.01)
                nn.init.constant_(m.bias, 0)