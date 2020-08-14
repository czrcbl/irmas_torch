import torch
import torchaudio
from torchvision.transforms import Compose
from torchaudio import transforms as atrans
import os
from os.path import join as pjoin
import numpy as np
import random


class Slice(torch.nn.Module):
    
    def __init__(self, time=1, fs=44100):
        super().__init__()
        self.time = time
        self.fs = fs
        
    def forward(self, x):
        ind = random.randint(0, int(x.shape[1] - self.time * self.fs) - 1)
        audio = x[:, ind: ind + int(self.time * self.fs)]
        
        return audio
    
class Normalize(torch.nn.Module):
    
    def __init__(self):
        super().__init__()
    
    def forward(self, x):
        x = x / torch.max(x, axis=1)[0]
        return x
 
   
class Mono(torch.nn.Module):
    
    def __init__(self):
        super().__init__()
    
    def forward(self, x):
        return x.mean(axis=0)[None, :]

class AsImageTrans:

    def __init__(self, mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)):
        self.mean = mean
        self.std = std
        self.norm = Normalize(mean=mean, std=std)

    def __call__(self, x):
        x = x.clamp(0, 1)
        x = x.repeat(3, 1, 1)
        x = self.norm(x)
        return x


class MelSpecTransformBase:
    """Base class to convert audio file to a Mel Spectogram."""
    def __init__(self, fs=22050, n_fft=1024, hop_length=256, n_mels=128, 
                 mono=True, orig_fs=44100, seed=101):
        self.fs = fs
        self.n_fft = n_fft
        self.hop_length = hop_length
        self.n_mels = n_mels
        self.mono = mono
        self.orig_fs = orig_fs
        random.seed(seed)

    def __call__(self, fn):
        pass

    def to_file(self, fn, output_path):
        out_spec = self(fn)
        out_fn =  pjoin(output_path, os.path.splitext(os.path.basename(fn))[0])
        if isinstance(out_spec, torch.Tensor):
            np.save(
                out_fn,
                out_spec.cpu().numpy().save())
        elif isinstance(out_spec, np.ndarray):
            np.save(out_fn, out_spec)


class MelSpecTransformTorchAudio(MelSpecTransformBase):
    """Transform audio with torchaudio"""
    def __init__(self, orig_fs=44100, **kargs):
        super().__init__(**kargs)
        self.orig_fs = orig_fs
        self.trans = [
            atrans.Resample(self.orig_fs, self.fs),
            atrans.MelSpectrogram(
                sample_rate=self.fs, 
                n_fft=self.n_fft, 
                hop_length=self.hop_length, 
                n_mels=self.n_mels)]

    def __call__(self, audio):
        if self.mono:
            audio = audio.mean(axis=0)[None, :]
        out = audio
        for trans in self.trans:
            out = trans(out)
        out = torch.log10(out)
        return out


