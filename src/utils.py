import torch
from torch import nn
import torchvision
from .networks import NetworkTop
from . import config as cfg


def get_network(base_network, transfer, mono):
    in_channels = 1 if mono else 2
#    if args.test:
#        net = TestNet(in_channels, 11).to(device)
    if base_network == 'resnet18':
        base_net = torchvision.models.resnet18(pretrained=transfer)
        last_channel = 512
    elif base_network == 'resnet34':
        base_net = torchvision.models.resnet34(pretrained=transfer)
        last_channel = 512
    elif base_network == 'resnet50':
        base_net = torchvision.models.resnet50(pretrained=transfer)
        last_channel = 2048
    elif base_network == 'resnet101':
        base_net = torchvision.models.resnet101(pretrained=transfer)
        last_channel = 2048
    
    if not transfer:
        base_net.conv1 = nn.Conv2d(in_channels, 64, kernel_size=(7, 7), stride=(2, 2))
    
    base_net = nn.Sequential(*list(base_net.children())[:-2])
    net = NetworkTop(base_net, len(cfg.classes), last_channel)
    
    return net

