import torch
from torch import nn
from torch.nn import functional as F
from torch import optim
from torch.utils import data as tdata
import torchvision
from torchsummary import summary
from ignite.metrics import Accuracy
import os
from os.path import join as pjoin
import sys
import logging
import argparse
from copy import deepcopy
import json
from tqdm import tqdm
from easydict import EasyDict as edict
import pickle
from src.datasets import IRMAS
from src.utils import get_network
from src import config as cfg


def parse_args():
    parser = argparse.ArgumentParser()
    # misc
    parser.add_argument('--dataset-name',type=str,
                        help='Dataset to use.')
    parser.add_argument('--model-name',type=str,
                        help='Experiment to use.')
    parser.add_argument('--strategy', type=str, default='1',
                        help='Evaluation strategy.')
#    parser.add_argument('--batch-size', type=int, default=1,
#                        help='Inference batch size.')
    parser.add_argument('--device',type=str, default='cuda',
                        help='Device to use.')
    parser.add_argument('--is-test', type=int, default=0,
                        help='Whether it is a test.')
    
    args = parser.parse_args()
    return args

def save_output(output, args, strategy):
    with open(pjoin(args.exp_path, f'output_strategy-{strategy}.pkl'), 'wb') as f:
        pickle.dump(output, f)
        
def test_strategy1(net, test_ds, device, args):
    
    net = net.to(device)
    output = []
    net.eval()
    with torch.no_grad():
        for data, target in tqdm(test_ds):
            data = data[None, :, :, :]
            data = data.to(device)
            y = net(data)
            y = F.sigmoid(y)
            output.append((y.cpu().numpy(), target.numpy()))
    
    save_output(output, args, 1)

        
        
def test_strategy2(net, test_ds, device, n_frames, args):
    
    net = net.to(device)
    output = []
    net.eval()
    with torch.no_grad():
        for data, target in tqdm(test_ds):
            data = data[None, :, :, :]
            data = data.to(device)
            ns = data.shape[-1] // n_frames
            out = []
            for i in range(ns):  
                y = net(data[:, :, :, i * n_frames: (i + 1) * n_frames])
                y = F.sigmoid(y)
                out.append(y.cpu().numpy())
#            y = net(data[:, :, :, (i + 1) * n_frames:])
#            y = F.sigmoid(y)        
#            out.append(y.cpu().numpy())
            
            output.append((out, target.numpy()))
            
    save_output(output, args, 2)
            
    

        
        
#def load_experiment(exp_name):
#    exp_path = pjoin(cfg.models_path, exp_name)
#    with open(pjoin(exp_path, 'parameters.json'), 'r') as f:
#        params = json.load(f)
#    audio_params = ['fs', 'n_fft', 'hop_length', 'n_mels']
#    audio_params = {key:params[key] for key in audio_params}
#    test_ds = IRMAS(mode='test', **audio_params)
#    net_params = edict({
#            'base_network': params['base_network'],
#            'transfer': False,
#            'mono':params['mono']})
#    net = get_network(net_params)
#    net.load_state_dict(torch.load(pjoin(exp_path, 'best_model.pth')))
#    return net, test_ds

def load_experiment(args):
    
    exp_path = pjoin(cfg.results_path, args.dataset_name, args.model_name)
    with open(pjoin(exp_path, 'parameters.json'), 'r') as f:
        params = json.load(f)
        
    audio_params = {key:params[key] for key in cfg.dataset_params}
    net_params = {key:params[key] for key in cfg.network_params}
    
    test_ds = IRMAS(mode='test', is_test=args.is_test, **audio_params)
    net = get_network(net_params['base_network'], False, net_params['mono'])
    net.load_state_dict(torch.load(pjoin(exp_path, 'best_model.pth')))
    args.exp_path = exp_path
    return net, test_ds, params

def main():
    
    args = parse_args()
    net, test_ds, params = load_experiment(args)
#    test_loader = tdata.DataLoader(test_ds, batch_size=args.batch_size)
    
    device = torch.device(args.device)
    
    ts = params['time_slice']
    hop_length = params['hop_length']
    fs = params['fs']
    n_frames = (fs * ts) // hop_length + 1
    
    if args.strategy == '1':
        test_strategy1(net, test_ds, device, args)
    elif args.strategy == '2':
        test_strategy2(net, test_ds, device, n_frames, args)
    elif args.strategy == 'all':
        test_strategy1(net, test_ds, device, args)
        test_strategy2(net, test_ds, device, n_frames, args)


if __name__ == '__main__':
    sys.exit(main())
    

