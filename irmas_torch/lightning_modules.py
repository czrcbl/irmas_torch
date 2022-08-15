import torch
from torch import nn
from torch.nn import functional as F
from torch.utils import data as tdata
import pytorch_lightning as pl
from irmas_torch import networks, datasets, transforms
from sklearn import metrics as skmetrics
import numpy as np
import torchmetrics
import hydra


class DataModule(pl.LightningDataModule):
    def __init__(self, *args, **kwargs):
        super().__init__()
        self.save_hyperparameters()

    def setup(self, stage):
        hp = self.hparams
        self.trn_trans = transforms.MelSpecTransformTorchAudio(**hp.transform)
        self.val_trans = transforms.MelSpecTransformTorchAudio(**hp.transform)
        self.test_trans = transforms.MelSpecTransformTorchAudio(**hp.transform)

        self.trn_ds = datasets.IRMASDataset(self.trn_trans, mode="train", **hp.dataset)
        self.val_ds = datasets.IRMASDataset(self.val_trans, mode="val", **hp.dataset)
        self.test_ds = datasets.IRMASDataset(self.test_trans, mode="test", **hp.dataset)
        print("train dataset", len(self.trn_ds))

    def train_dataloader(self):
        hp = self.hparams
        return tdata.DataLoader(self.trn_ds, **hp.dataloader.train)

    def val_dataloader(self):
        hp = self.hparams
        return tdata.DataLoader(self.val_ds, **hp.dataloader.train)

    def test_dataloader(self):
        hp = self.hparams
        return tdata.DataLoader(self.test_ds, **hp.dataloader.train)


class ResnetIrmas(pl.LightningModule):
    def __init__(self, *args, **kargs):
        super().__init__()
        self.save_hyperparameters()
        self.lr = self.hparams.module.optimizer.lr

        self.criterion = nn.BCEWithLogitsLoss()
        # self.net = networks.Resnet18(outdim=11)
        self.net = hydra.utils.instantiate(self.hparams.module.model)

        # Metrics
        self.val_recall = torchmetrics.Recall(multilabel=True, threshold=0.5)
        self.val_precision = torchmetrics.Precision(multilabel=True, threshold=0.5)
        self.val_accuracy = torchmetrics.Recall(multilabel=True, threshold=0.5)
        self.val_f1 = torchmetrics.F1Score(multilabel=True, threshold=0.5)

    def forward(self, x):
        return self.net(x)

    def configure_optimizers(self):

        optimizer = hydra.utils.instantiate(
            self.hparams.module.optimizer, self.parameters(), lr=self.lr
        )
        # optimizer = torch.optim.Adam(
        #     self.parameters(), self.lr, **self.hparams.module.optimizer
        # )

        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
            optimizer,
            mode="min",
            factor=0.1,
            patience=self.hparams.scheduler_patience,
            threshold=0.0001,
            threshold_mode="rel",
            cooldown=0,
            min_lr=0,
            eps=1e-08,
            verbose=True,
        )
        return {
            "optimizer": optimizer,
            "lr_scheduler": scheduler,
            "monitor": "val_loss",
        }

    def training_step(self, batch, batch_id):

        x, y = batch
        output = self(x)
        loss = F.binary_cross_entropy_with_logits(output, y)
        return loss

    def validation_step(self, batch, batch_idx):

        x, y = batch
        output = self(x)
        loss = F.binary_cross_entropy_with_logits(output, y)
        y_pred = torch.sigmoid(output)

        self.val_recall(y_pred, y.to(torch.int))
        self.val_precision(y_pred, y.to(torch.int))
        self.val_accuracy(y_pred, y.to(torch.int))
        self.val_f1(y_pred, y.to(torch.int))

        # Logging
        self.log("val_loss", loss.item(), prog_bar=True)
        self.log(
            "val_recall",
            self.val_recall,
            prog_bar=True,
            logger=True,
            on_epoch=True,
            on_step=False,
        )
        self.log(
            "val_precision",
            self.val_precision,
            prog_bar=True,
            logger=True,
            on_epoch=True,
            on_step=False,
        )
        self.log(
            "val_accuracy",
            self.val_accuracy,
            prog_bar=True,
            logger=True,
            on_epoch=True,
            on_step=False,
        )
        self.log(
            "val_f1",
            self.val_f1,
            prog_bar=True,
            logger=True,
            on_epoch=True,
            on_step=False,
        )

        outputs = {"loss": loss, "y_pred": y_pred.cpu(), "y": y.cpu()}
        return outputs

    def validation_epoch_end(self, outputs):
        y_pred = []
        y = []
        for out in outputs:
            y_pred.append(out["y_pred"].numpy())
            y.append(out["y"].numpy())

        y_pred = np.vstack(y_pred)
        y = np.vstack(y)

        mAP = skmetrics.average_precision_score(y, y_pred)

        # self.log('val_map', mAP, prog_bar=True, logger=True, on_step=False, on_epoch=True)
        self.log("val_mAP", mAP)
