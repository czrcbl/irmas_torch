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


class IRMASDataset(tdata.Dataset):
    """"""

    def __init__(
        self, transform, root, mode="train", train_fraq=0.9, seed=101, **kwargs
    ):
        super().__init__()
        self.root = root
        self.mode = mode
        self.train_fraq = train_fraq
        self.seed = seed
        self.fns, self.labels = self.load_files()
        self.classes = cfg.classes
        self.clsarr = np.array(self.classes)
        self.trans = transform

    def __len__(self):
        return len(self.fns)

    def load_trainset(self):

        labels = []
        fns = []
        trn_root = pjoin(self.root, "IRMAS-TrainingData")
        classes = sorted(
            [d for d in os.listdir(trn_root) if os.path.isdir(pjoin(trn_root, d))]
        )
        for _cls in classes:
            path = pjoin(trn_root, _cls)
            files = sorted(os.listdir(path))
            fns.extend([pjoin(path, f) for f in files])
            labels.extend([_cls] * len(files))
        idxs = list(range(len(fns)))
        random.seed(self.seed)
        random.shuffle(idxs)
        fns = [fns[i] for i in idxs]
        labels = [labels[i] for i in idxs]
        if self.mode == "train":
            fns = fns[: int(self.train_fraq * len(fns))]
            labels = labels[: int(self.train_fraq * len(labels))]
        elif self.mode == "val":
            fns = fns[int(self.train_fraq * len(fns)) :]
            labels = labels[int(self.train_fraq * len(labels)) :]

        # print(fns, labels)
        return fns, labels

    def load_testset(self):

        fns, labels = [], []
        test_root = pjoin(self.root, "IRMAS-TestingData")
        for folder in sorted(os.listdir(test_root)):
            folder_path = pjoin(test_root, folder)
            files = sorted(os.listdir(folder_path))
            audio_files = [
                pjoin(folder_path, f)
                for f in files
                if os.path.splitext(f)[-1] == ".wav"
            ]
            for af in audio_files:
                labels_file = pjoin(folder_path, os.path.splitext(af)[0] + ".txt")
                with open(labels_file, "r") as f:
                    lines = [l.strip() for l in f.readlines()]
                labels.append(lines)
                fns.append(pjoin(folder_path, af))
        return fns, labels

    def load_files(self):
        if self.mode in ["train", "val"]:
            return self.load_trainset()
        elif self.mode == "test":
            return self.load_testset()

    def load_audio(self, idx):
        fn = self.fns[idx]
        label = self.labels[idx]
        audio, orig_fs = torchaudio.load(fn)
        return audio, orig_fs, label

    def __getitem__(self, idx):

        audio, orig_fs, label = self.load_audio(idx)
        if isinstance(label, list):
            label_arr = np.zeros(
                (
                    len(
                        self.classes,
                    )
                )
            )
            for item in list(set(label)):
                label_arr += (self.clsarr == item).astype("int")
        else:
            label_arr = (self.clsarr == label).astype("int")

        out = self.trans(audio)

        return out, torch.tensor(label_arr, dtype=torch.float32)
