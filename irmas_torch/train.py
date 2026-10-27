import os
from os.path import join as pjoin
import logging
from copy import deepcopy
from tqdm import tqdm

import numpy as np
from sklearn.metrics import average_precision_score, f1_score

import torch
from torch import optim
from ignite import metrics

from . import config as cfg
from . import transforms


def start_logger(root, name):
    logging.basicConfig()
    logger = logging.getLogger()
    logger.setLevel(logging.INFO)
    log_file_path = pjoin(root, f'{name}.log')
    # log_dir = os.path.dirname(log_file_path)
    fh = logging.FileHandler(log_file_path)
    logger.addHandler(fh)
    
    return logger

class ModelSaver:
    
    def __init__(self, save_path):
        self.best_score = 0 
        self.best_model = None
        self.save_path = save_path
        
    def update(self, net, score):
        if score > self.best_score:
            self.best_score = score
            self.best_params = deepcopy(net.state_dict())
            
    def close(self):
        torch.save(self.best_params, pjoin(self.save_path, 'best_model.pth'))
        
        
class Trainer:
    
    def __init__(self, root, model, trn_loader, val_loader, criterion, 
                 optimizer=None, scheduler=None, device='cuda'):
        
        self.root = root
        # try:
        #     os.makedirs(root)
        # except OSError:
        #     raise ValueError('Experiment Already Exists!')
        self.device = device
        self.model = model.to(device)
        self.trn_loader = trn_loader
        self.val_loader = val_loader
        self.criterion = criterion
        if optimizer is None:
            self.optimizer = optim.Adam(self.model.parameters())
        else:
            self.optimizer = optimizer
            
        if scheduler is None:
            self.scheduler = optim.lr_scheduler.StepLR(self.optimizer, step_size=1, gamma=0.95)
        else:
            self.scheduler = scheduler

        # self.trn_metrics = {'accu': Accuracy()}
        #                     # 'CELoss': CELossMetric()}
        # self.val_metrics = {'accu': Accuracy()}
        #                     # 'CELoss': CELossMetric()}
                        
        
        
        self.curr_epoch = 0
        self.logger = start_logger(self.root, 'train')
        self.saver = ModelSaver(self.root)
        
        
    
    def train(self, epochs):
    
        net = self.model
        trn_loader = self.trn_loader
        val_loader = self.val_loader
        optimizer = self.optimizer
        criterion = self.criterion
        device = self.device
        saver = self.saver
        logger = self.logger
        
        accu_metric = metrics.Accuracy()

        for epoch in range(epochs):
            net.train()
            train_loss = 0
            logger.info(f'Epoch {epoch} training starting.')
            for i, (data, target) in tqdm(enumerate(trn_loader)):
                data, target = data.to(device), target.to(device)
                optimizer.zero_grad()
                y = net(data)
                loss = criterion(y, target.float())
                loss.backward()
                optimizer.step()
                train_loss += loss.item()
            train_loss /= i
            
            accu=0
            net.eval()
            y_true = []
            y_pred = []
            # accu_metric.reset()
            with torch.no_grad():
                val_loss = 0
                logger.info(f'Epoch {epoch} validation starting.')
                for i, (data, target) in tqdm(enumerate(val_loader)):
                    data, target = data.to(device), target.to(device)
                    y = net(data)
                    loss = criterion(y, target.float())
                    val_loss += loss.item()
                    y = torch.sigmoid(y)
                    y_pred.append(y.cpu().numpy())
                    y_true.append(target.cpu().numpy())
                    accu_metric.update((y.round(), target))
                    accu += torch.mean((y.argmax(axis=-1) == target.argmax(axis=-1)).float())
                val_loss /= i
                # TODO: Proper weight the accuracy
                accu /= i
            y_true = np.vstack(y_true)
            y_pred = np.vstack(y_pred)
            mAP = average_precision_score(y_true, y_pred)
            f1score = f1_score(y_true, np.round(y_pred), average='weighted')
            msg = f'Epoch: {epoch}\n'\
                f'train_loss: {train_loss:.3f}\n'\
                f'validation_loss: {val_loss:.3f}\n'\
                f'single_class_accuracy: {accu:.3f}\n'\
                f'multiclass_class_accuracy: {accu_metric.compute():.3f}\n'\
                f'F1-score: {f1score:.3f}\n'\
                f'mAP {mAP:.3f}'
            logger.info(msg)
            saver.update(net, accu)
        
        saver.close()