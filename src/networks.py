import torch
from torch import nn
from torch.nn import functional as F


class TestNet(nn.Module):
    
    def __init__(self, n_channels, n_classes):
        super().__init__()
        
        self.conv0 = nn.Conv2d(n_channels, 32, 3, padding=1)
        self.relu0 = nn.ReLU()
        self.maxpool0 = nn.MaxPool2d(2)
        self.conv1 = nn.Conv2d(32, 64, 3, padding=1)
        self.relu1 = nn.ReLU()
        self.maxpool1 = nn.MaxPool2d(2)
        
        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        
        self.linear0 = nn.Linear(64, n_classes)
        
    def forward(self, x):
        
        x = self.conv0(x)
        x = self.relu0(x)
        x = self.maxpool0(x)
        x = self.conv1(x)
        x = self.relu1(x)
        x = self.maxpool1(x)
        
        x = self.avgpool(x)
        x = nn.Flatten()(x)
        x = self.linear0(x)
        
        return x
    
class NetworkTop(nn.Module):
    
    def __init__(self, base, n_classes, last_channel):
        super().__init__()
        self.base = base
        self.avgpool = nn.AdaptiveAvgPool2d((1,1))
        self.flatten = nn.Flatten()
        self.linear = nn.Linear(last_channel, n_classes)
    
    def forward(self, x):
        x = self.base(x)
        x = self.avgpool(x)
        x = self.flatten(x)
        x = self.linear(x)
        
        return x


if __name__ == '__main__':

    from torchsummary import summary
    import torchvision
    device = torch.device('cpu')
    base_net = torchvision.models.resnet34(pretrained=False)
    summary(base_net, (3, 128, 87), device='cpu')
    base_net.conv1 = nn.Conv2d(1, 64, kernel_size=(7, 7), stride=1)
    net = NetworkTop(base_net, 11, 512)
    
        
        

