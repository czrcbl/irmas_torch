from gc import callbacks
import pytorch_lightning as pl
import hydra
from omegaconf import DictConfig
from irmas_torch import lightning_modules as lm


@hydra.main(config_path="config", config_name="config")
def main(cfg: DictConfig):
    model = lm.ResnetIrmas(**cfg)
    logger = pl.loggers.TensorBoardLogger(".", name=cfg.experiment_name)

    callbacks = [
        pl.callbacks.early_stopping.EarlyStopping(
            monitor="val_loss",
            min_delta=0.00,
            patience=cfg.early_stopping_patience,
            verbose=False,
            mode="min",
        ),
        pl.callbacks.ModelCheckpoint(
            dirpath=f"{logger.save_dir}/checkpoints",
            monitor="val_loss",
        ),
        pl.callbacks.LearningRateMonitor(),
    ]

    data_module = lm.DataModule(**cfg)

    trainer = pl.Trainer(
        gpus=1,
        logger=logger,
        max_epochs=cfg.max_epochs,
        callbacks=callbacks,
    )
    trainer.fit(model, train_dataloaders=data_module)


if __name__ == "__main__":

    main()
