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
from .transforms import MelSpecTransformTorchAudio


class IRMAS(tdata.Dataset):
    
    def __init__(self, root=cfg.irmas_path, mode='train',
                 train_fraq=0.8,
                 fs=22050, n_fft=1024, hop_length=256, n_mels=128, 
                 time_slice=None, mono=True, normalize=True, 
                 preprocess=False, seed=101, 
                 is_test=False):
        super().__init__()
        self.root = root
        self.mode = mode
        self.train_fraq = train_fraq
        self.fs = 22050
        self.n_fft = n_fft
        self.hop_length = hop_length
        self.n_mels = n_mels
        self.normalize = normalize
        self.time_slice = time_slice
        self.mono = mono
        self.seed = seed
        
        fns, labels = self.load_fns()
        if is_test:
            random.seed(seed)
            fns = random.sample(fns, 55)
            labels = random.sample(labels, 55)
            
        self.fns, self.labels = fns, labels
        self.classes = cfg.classes
        self.clsarr = np.array(self.classes)
        
        self.trans = MelSpecTransformTorchAudio(
                fs=self.fs, n_fft=self.n_fft, hop_length=self.hop_length, 
                n_mels=self.n_mels, mono=self.mono, time_slice=self.time_slice, 
                seed=self.seed)

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
            random.seed(self.seed)
            idxs = list(range(len(fns)))
            random.shuffle(idxs)
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
                
        
    def __len__(self):
        return len(self.fns)
    
    
    def __getitem__(self, idx):
        fn = self.fns[idx]
        label = self.labels[idx]
        mel_spec = self.trans(fn)
        mel_spec = torch.log10(mel_spec)
        if self.normalize:
#            mel_spec = mel_spec - mel_spec.mean(axis=(-2, -1))[:, None, None] \
#            / mel_spec.std(axis=(-2, -1))[:, None, None]
            mel_spec = torch.clamp(mel_spec, min=-1, max=1)
        if isinstance(label, list):
            label_arr = np.zeros((len(self.classes,)))
            for item in list(set(label)):
                label_arr += (self.clsarr == item).astype('int')
        else:
            label_arr = (self.clsarr == label).astype('int')
        
        return mel_spec.detach(), torch.tensor(label_arr)
        
    
    def get_audio(self, idx, numpy=False):
        audio, fs = torchaudio.load(self.fns[idx])
        if numpy:
            audio = audio.numpy()
        return audio, fs
    
    def preprocess_dataset(self):
        pass
        
        
    
    

