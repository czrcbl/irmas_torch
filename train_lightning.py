import torch
from torch import nn
from torch.nn import functional as F
from torch.utils import data as tdata
import pytorch_lightning as pl
from src import networks, datasets, transforms
import hydra
from omegaconf import DictConfig
from sklearn import metrics
import numpy as np


@hydra.main(config_path='config', config_name='config')
def main(cfg: DictConfig):  
    model = IrmasModule(**cfg)
    logger = pl.loggers.TensorBoardLogger('.')
    trainer = pl.Trainer(gpus=1, logger=logger, max_epochs=cfg.max_epochs, profiler=True, weights_summary='full')
    trainer.fit(model)
    
    
class IrmasModule(pl.LightningModule):
    
    def __init__(self, *args, **kargs):
        super().__init__()
        self.save_hyperparameters()
        self.lr = self.hparams.lr
        
        self.criterion = nn.BCEWithLogitsLoss()
        self.net = networks.Resnet18(outdim=11)
        
        # Metrics
        self.val_recall = pl.metrics.Recall(multilabel=True)
        self.val_precision = pl.metrics.Precision(multilabel=True)
        self.val_accuracy = pl.metrics.Recall(multilabel=True)
        
    def forward(self, x):
        return self.net(x)
        
    def setup(self, stage):
        hp = self.hparams
        self.trn_trans = transforms.MelSpecTransformTorchAudio(**hp)
        self.val_trans = transforms.MelSpecTransformTorchAudio(**hp)
        self.test_trans = transforms.MelSpecTransformTorchAudio(**hp)
        
        self.trn_ds = datasets.IRMAS(self.trn_trans, root=hp.dataset_path, mode='train') 
        self.val_ds = datasets.IRMAS(self.val_trans, root=hp.dataset_path, mode='val')
        self.test_ds = datasets.IRMAS(self.test_trans, root=hp.dataset_path, mode='test')  
        
    def train_dataloader(self):
        return tdata.DataLoader(self.trn_ds, 32, num_workers=8)
    
    def val_dataloader(self):
        hp = self.hparams
        return tdata.DataLoader(self.val_ds, hp.batch_size, num_workers=hp.num_workers)
    
    
    def test_dataloader(self):
        hp = self.hparams
        return tdata.DataLoader(self.test_ds, hp.batch_size, num_workers=hp.num_workers)
    
    def configure_optimizers(self):
        optimizer = torch.optim.Adam(self.parameters(), self.lr)
        return optimizer
    
    def training_step(self, batch, batch_id):
        
        x, y = batch
        output = self(x)
        loss = F.binary_cross_entropy_with_logits(output, y)
        # loss = self.criterion(output, y)
        return loss
    
    def validation_step(self, batch, batch_idx):
        
        x, y = batch
        output = self(x)
        loss = F.binary_cross_entropy_with_logits(output, y)
        # loss = self.criterion(output, y)
        y_pred = torch.sigmoid(output)
        
        self.val_recall(y_pred, y)
        self.val_precision(y_pred, y)
        self.val_accuracy(y_pred, y)
        
        # Logging
        self.log('val_recall', self.val_recall, prog_bar=True, logger=True, on_epoch=True, on_step=False)
        self.log('val_precision', self.val_precision, prog_bar=True, logger=True, on_epoch=True, on_step=False)
        self.log('val_accuracy', self.val_accuracy, prog_bar=True, logger=True, on_epoch=True, on_step=False)
        
        outputs = {
            'loss': loss,
            'y_pred': y_pred.cpu(),
            'y': y.cpu()
        }
        return outputs
    
    def validation_epoch_end(self, outputs):
        y_pred = []
        y = []
        for out in outputs:
            y_pred.append(out['y_pred'].numpy())
            y.append(out['y'].numpy())

        y_pred = np.vstack(y_pred)
        y = np.vstack(y)
        
        mAP = metrics.average_precision_score(y, y_pred)
        
        # self.log('val_map', mAP, prog_bar=True, logger=True, on_step=False, on_epoch=True)
        
        log = {
            'mAP': mAP,
            'global_step': self.current_epoch
        }
        
        results = {
            'log': log
        }
        
        return results
        
if __name__ == '__main__':
    
    main()
        
        