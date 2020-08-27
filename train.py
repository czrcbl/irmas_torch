import argparse
import os
from os.path import join as pjoin
from easydict import EasyDict as edict
import yaml 
try:
    from yaml import CLoader as Loader
except ImportError:
    from yaml import Loader

from torch import nn
import torchvision
from torch.utils import data as tdata
from torchvision.transforms import Compose
from torchaudio import transforms as atrans

from src import networks
from src.train import Trainer
from src.datasets import IRMAS
from src import config as cfg
from src import transforms as trans


def get_transfroms(params):
    
    translist = [
        trans.Slice(time=params.transform.time_slice),
        atrans.Resample(params.transform.orig_fs, new_freq=params.transform.fs),
    ]
    if params.transform.mono:
        translist.append(trans.Mono())
    if params.transform.normalize:
        translist.append(trans.Normalize())
    
    if params.usemelspec:
        translist.append(
            atrans.MelSpectrogram(
                sample_rate=params.transform.fs, 
                n_fft=params.spec_transform.n_fft, 
                hop_length=params.spec_transform.hop_length, 
                n_mels=params.spec_transform.n_mels)
        )
        
    return Compose(translist)


def get_datasets(transforms, params):
    
    if params.dataset.name.lower() == 'irmas':
        trn_ds = IRMAS(transforms, root=params.dataset.root, mode='train', train_fraq=params.dataset.train_fraq, seed=params.dataset.seed)
        val_ds = IRMAS(transforms, root=params.dataset.root, mode='val', train_fraq=params.dataset.train_fraq, seed=params.dataset.seed)
    
    return trn_ds, val_ds


def get_dataloaders(trn_ds, val_ds, params):
    trn_loader = tdata.DataLoader(
        trn_ds, batch_size=params.train.batch_size, num_workers=params.train.num_workers)
    val_loader = tdata.DataLoader(
        val_ds, batch_size=params.train.batch_size, num_workers=params.train.num_workers)
    return trn_loader, val_loader


def get_model(params):
    in_channels = 1 if params.transform.mono else 2
#    if args.test:
#        net = TestNet(in_channels, 11).to(device)
    if params.model.name == 'resnet18':
        base_net = torchvision.models.resnet18(pretrained=params.model.transfer)
        last_channel = 512
    elif params.model.name == 'resnet34':
        base_net = torchvision.models.resnet34(pretrained=params.model.transfer)
        last_channel = 512
    elif params.model.name == 'resnet50':
        base_net = torchvision.models.resnet50(pretrained=params.model.transfer)
        last_channel = 2048
    elif params.model.name == 'resnet101':
        base_net = torchvision.models.resnet101(pretrained=params.model.transfer)
        last_channel = 2048
    
    if not params.model.transfer:
        base_net.conv1 = nn.Conv2d(in_channels, 64, kernel_size=(7, 7), stride=(2, 2))
    
    base_net = nn.Sequential(*list(base_net.children())[:-2])
    net = networks.NetworkTop(base_net, len(cfg.classes), last_channel)
    
    return net

def parse_args():
    parser = argparse.ArgumentParser()
    # misc
    parser.add_argument('--is-test',type=int, default=0,
                        help='Whether it is a test.')
    parser.add_argument('--expfile', type=str, help='Path to config file describing the experiment.')
    args = parser.parse_args()

    return args


def main():
    args = parse_args()
    with open(args.expfile, 'r') as f:
        params_dict = yaml.load(f, Loader=Loader)
        params = edict(params_dict)
    
    try:
        os.makedirs(params.root)
    except OSError:
        raise ValueError('Root folder already exists.')
    
    with open(pjoin(params.root, 'expfile.yaml'), 'w') as f:
        yaml.dump(params, f)
        
    transforms = get_transfroms(params)
    trn_ds, val_ds = get_datasets(transforms, params)
    trn_loader, val_loader = get_dataloaders(trn_ds, val_ds, params)
    model = get_model(params)
    criterion = nn.BCEWithLogitsLoss()
    trainer = Trainer(params.root, model, trn_loader, val_loader, criterion=criterion, device=params.train.device)
    trainer.train(params.train.epochs)

if __name__ == '__main__':
    main()