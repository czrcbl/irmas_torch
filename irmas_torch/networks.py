import math
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
    
class RConv(nn.Module):
    
    def __init__(self, in_channels, out_channels, kernel_size, stride, padding='same', *args, **kargs):
        super().__init__(*args, **kargs)
        
        # padding chosen in order to mantain size if no strida, ou reduce size to input_dim/stride
        self.kernel_size = kernel_size
        self.stride = stride
        try:
            self.kh = self.kernel_size[0] # height dim
            self.kw = self.kernel_size[1] # width dim
        except (IndexError, TypeError):
            self.kh = self.kw = self.kernel_size
        try:
            self.sh = self.stride[0] # height dim
            self.sw = self.stride[1] # width dim
        except (IndexError, TypeError):
            self.sh = self.sw = self.stride
            
        self.conv0 = nn.Conv2d(in_channels=in_channels, out_channels=out_channels, kernel_size=kernel_size, stride=stride)
        self.bnorm0 = nn.BatchNorm2d(out_channels)
        self.relu0 = nn.ReLU()
        
    def forward(self, x, residual=None):
        
        kw, kh, sw, sh = self.kw, self.kh, self.sw, self.sh
        Niw = x.shape[-1]
        Nih = x.shape[-2]
        Now = math.ceil(Niw / sw)
        Noh = math.ceil(Nih / sh) 
        pw = max(sw * (Now - 1) - Niw + kw, 0)
        ph = max(sh * (Noh - 1) - Nih + kh, 0)
        padl =  pw // 2
        padr = pw - padl
        padt =  ph // 2
        padb = ph - padt
        x = F.pad(x, (padl, padr, padt, padb), value=0)
        x = self.conv0(x)
        x = self.bnorm0(x)
        if residual is not None:
            assert(x.shape == residual.shape)
            x = x + residual
        x = self.relu0(x)
        
        return x


class ResidualBlock(nn.Module):
    
    def __init__(self, in_channels, out_channels, kernel_size, stride, *args, **kargs):
        super().__init__(*args, **kargs)
        
        if stride != 1 and stride != (1, 1):
            self.pre = nn.Conv2d(in_channels=in_channels, out_channels=out_channels, kernel_size=(1, 1), stride=stride)
        else:
            self.pre = nn.Identity()
        self.rconv0 = RConv(in_channels, out_channels, kernel_size, stride)
        self.rconv1 = RConv(out_channels, out_channels, kernel_size, stride=(1, 1))
        
    def forward(self, x):
        
        x1 = self.pre(x)
        x = self.rconv0(x)
        x = self.rconv1(x, residual=x1)
        
        return x
    

class BottleneckBlock(nn.Module):
    
    def __init__(self, in_channels, out_channels, kernel_size, stride, *args, **kargs):
        super().__init__(*args, **kargs)
        
        if stride != 1 and stride != (1, 1):
            self.pre = nn.Conv2d(in_channels=in_channels, out_channels=4 * out_channels, kernel_size=(1, 1), stride=stride)
        else:
            self.pre = nn.Identity()
        self.rconv0 = RConv(in_channels, out_channels, kernel_size=(1, 1), stride=stride)
        self.rconv1 = RConv(out_channels, out_channels, kernel_size, stride=(1, 1))
        self.rconv2 = RConv(out_channels, 4 * out_channels, kernel_size=(1, 1), stride=(1, 1))
        
    def forward(self, x):
        
        x1 = self.pre(x)
        x = self.rconv0(x)
        x = self.rconv1(x)
        x = self.rconv2(x, residual=x1)
        
        return x
    
class MultiBlock(nn.Module):

    def __init__(self, in_channels, out_channels, kernel_size, strides, size, *args, blocktype='residual', **kargs):
        super().__init__(*args, **kargs)
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.kernel_size = kernel_size
        self.strides = strides
        self.size = size
        self.blocktype = blocktype

        if blocktype == 'residual':
            self.cons = ResidualBlock
        elif blocktype == 'bottleneck':
            self.cons = BottleneckBlock
        
        self.blocks = []
        for i in range(self.size):
            if i == 0:
                self._modules[f'block{i:03d}'] = self.cons(in_channels, out_channels, kernel_size, strides, *args, **kargs)
            else:
                self._modules[f'block{i:03d}'] = self.cons(out_channels, out_channels, kernel_size, (1, 1), *args, **kargs)
            
            # self.blocks.append(self._modules[f'block{i:03d}'])
    
    def forward(self, x):

        for block in self._modules.values():
            x = block(x)

        return x


class Resnet18(nn.Module):
    
    def __init__(self, in_channels=1, outdim=2, *args, **kargs):
        super().__init__(*args, **kargs)
        self.input_conv = nn.Conv2d(in_channels, 64, kernel_size=(7, 7), stride=(2, 2), padding=(3, 3))
        self.maxpool = nn.MaxPool2d(kernel_size=(3, 3), stride=(2, 2))
        
        self.rb00 = ResidualBlock(64, 64, (3, 3), stride=1)
        self.rb01 = ResidualBlock(64, 64, (3, 3), stride=1)
        
        self.rb10 = ResidualBlock(64, 128, (3, 3), stride=2)
        self.rb11 = ResidualBlock(128, 128, (3, 3), stride=1)
        
        self.rb20 = ResidualBlock(128, 256, (3, 3), stride=2)
        self.rb21 = ResidualBlock(256, 256, (3, 3), stride=1)
        
        self.rb30 = ResidualBlock(256, 512, (3, 3), stride=2)
        self.rb31 = ResidualBlock(512, 512, (3, 3), stride=1)
        
        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        
        self.flatten = nn.Flatten()
        
        self.linear = nn.Linear(512, outdim)
        
    def forward(self, x):
        
        x = self.input_conv(x)
        x = self.maxpool(x)
        
        x = self.rb00(x)
        x = self.rb01(x)
        
        x = self.rb10(x)
        x = self.rb11(x)
        
        x = self.rb20(x)
        x = self.rb21(x)
        
        x = self.rb30(x)
        x = self.rb31(x)
        
        x = self.avgpool(x)
        x = self.flatten(x)
        x = self.linear(x)
        
        return x
    
    
class Resnet34(nn.Module):

    def __init__(self, in_channels=1, outdim=2, *args, **kargs):
        super().__init__(*args, **kargs)
        self.outdim = outdim

        self.inconv = nn.Conv2d(in_channels, 64, (7, 7), (2, 2), padding=(3, 3))
        
        self.inmaxpool = nn.MaxPool2d(kernel_size=(3, 3), stride=(2, 2), padding=(1, 1))

        self.block0 = MultiBlock(64, 64, (3, 3), (1, 1), 3, blocktype='residual')

        self.block1 = MultiBlock(64, 128, (3, 3), (2, 2), 4, blocktype='residual')

        self.block2 = MultiBlock(128, 256, (3, 3), (2, 2), 6, blocktype='residual')

        self.block3 = MultiBlock(256, 512, (3, 3), (2, 2), 3, blocktype='residual')

        self.pool = nn.AdaptiveAvgPool2d((1, 1))
        self.flatten = nn.Flatten()
        self.out = nn.Linear(512, outdim)

    def forward(self, x):

        x = self.inconv(x)
        x = self.inmaxpool(x)
        x = self.block0(x)
        x = self.block1(x)
        x = self.block2(x)
        x = self.block3(x)
        x = self.pool(x)
        x = self.flatten(x)
        x = self.out(x)

        return x


if __name__ == '__main__':

    from torchsummary import summary
    import torchvision
    device = torch.device('cpu')
    base_net = torchvision.models.resnet34(pretrained=False)
    summary(base_net, (3, 128, 87), device='cpu')
    base_net.conv1 = nn.Conv2d(1, 64, kernel_size=(7, 7), stride=1)
    net = NetworkTop(base_net, 11, 512)
    
        
        

