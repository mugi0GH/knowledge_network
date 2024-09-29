# from torchvision.models import resnet50, ResNet50_Weights

from torchvision.models import vgg16, VGG16_Weights,VGG

from torch import nn
import torch
from torchvision import transforms

from PIL import Image

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# image = Image.open('./grace_hopper_517x606.jpg')
# Define a transform to convert the image to a tensor
# transform = transforms.ToTensor()
# normalize = transforms.Normalize(
#     mean=[0.4914, 0.4822, 0.4465],
#     std=[0.2023, 0.1994, 0.2010],
# )
# transform = transforms.Compose([
#         transforms.ToTensor(),
#         normalize,
# ])
# Apply the transform to convert the image into a tensor
# tensor = transform(image).unsqueeze(0).to(device)

tensor = torch.randn(1, 3, 650, 550).to(device)  # Example input with batch_size=1, channels=512, height=32, width=32


from torchvision.ops import roi_pool
class backbone(nn.Module):
    def __init__(self, init_weights=True):
        super(backbone, self).__init__()
        # Load the pre-trained VGG model
        self.vgg = vgg16(weights=VGG16_Weights.DEFAULT)
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

model = backbone(init_weights=True).to(device)  # For a problem with 10 classes



# put it into model, add classifer and regressor


rois = torch.tensor([[0, 1, 1, 10, 10], [0, 5, 10, 15, 15],[0, 5, 10, 20, 15]], dtype=torch.float).to(device) # batch index,(x1, y1, x2, y2)
output_size = (7, 7)
pooled_output = roi_pool(input = model(tensor),boxes= rois,output_size= output_size,spatial_scale=1.0)
flat = nn.Flatten()
x = flat(pooled_output)
print(x.shape[1])
Linear1 = nn.Linear(in_features=x.shape[1],out_features=4096).to(device)
Linear2 = nn.Linear(in_features=4096,out_features=4096).to(device)
x = Linear1(x)
x = Linear2(x)