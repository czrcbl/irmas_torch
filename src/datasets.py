import torch
from torch.utils import data as tdata
import torchaudio
from torchaudio import transforms as tatrans
from torchvision import transforms as tvtrans
import os
from os.path import join as pjoin
import random
from copy import copy
import numpy as np
from . import config as cfg
from .transforms import MelSpecTransformTorchAudio, Compose
    
class IRMAS(tdata.Dataset):
    
    def __init__(self, transform, root, mode='train', train_fraq=0.8, seed=101):
        super().__init__()
        self.root = root
        self.mode = mode
        self.train_fraq = train_fraq
        self.seed = seed
        fns, labels = self.load_fns() 
        self.fns, self.labels = fns, labels
        self.classes = cfg.classes
        self.clsarr = np.array(self.classes)
        self.trans = transform
        
    def __len__(self):
        return len(self.fns)
        
    def load_fns(self):
        labels = []
        fns = []
        if (self.mode == 'train') or (self.mode == 'val'):
            trn_root = pjoin(self.root, 'IRMAS-TrainingData')
            classes = sorted([d for d in os.listdir(trn_root) if os.path.isdir(pjoin(trn_root, d))])
            for _cls in classes:
                path = pjoin(trn_root, _cls)
                files = sorted(os.listdir(path)) 
                fns.extend([pjoin(path, f) for f in files])
                labels.extend([_cls]*len(files))
            idxs = list(range(len(fns)))
            state = random.getstate()
            random.seed(self.seed)
            random.shuffle(idxs)
            random.setstate(state)
            fns = [fns[i] for i in idxs]
            labels = [labels[i] for i in idxs]
            if self.mode == 'train':
                fns = fns[:int(self.train_fraq * len(fns))]
                labels = labels[:int(self.train_fraq * len(labels))]
            elif self.mode == 'val':
                fns = fns[int(self.train_fraq * len(fns)):]
                labels = labels[int(self.train_fraq * len(labels)):]

        elif self.mode == 'test':
            test_root = pjoin(self.root, 'IRMAS-TestingData')
            for folder in sorted(os.listdir(test_root)):
                folder_path = pjoin(test_root, folder)
                files = sorted(os.listdir(folder_path))
                audio_files = [pjoin(folder_path, f) for f in files if os.path.splitext(f)[-1] == '.wav']
                for af in audio_files:
                    labels_file = pjoin(folder_path, os.path.splitext(af)[0] + '.txt')
                    with open(labels_file, 'r') as f:
                        lines = [l.strip() for l in f.readlines()]
                    labels.append(lines)
                    fns.append(pjoin(folder_path, af))   
        return fns, labels
    
    def __getitem__(self, idx):
        
        fn = self.fns[idx]
        label = self.labels[idx]
        audio, orig_fs = torchaudio.load(fn)
        
        if isinstance(label, list):
            label_arr = np.zeros((len(self.classes,)))
            for item in list(set(label)):
                label_arr += (self.clsarr == item).astype('int')
        else:
            label_arr = (self.clsarr == label).astype('int')
            
        out = self.trans(audio)
        
        return out, torch.tensor(label_arr, dtype=torch.float32)