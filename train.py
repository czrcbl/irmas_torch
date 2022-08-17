import pytorch_lightning as pl
import hydra
from omegaconf import DictConfig
from irmas_torch import lightning_modules as lm



# def callback_function(operation_type, model_info):
#     """Sample model info:
#     {'model': None,
#     'upload_filename': 'epoch=0-step=755.ckpt',
#     'local_model_path': '/home/cezar/Projects/irmas_torch/outputs/2022-08-15/01-37-08/default/checkpoints/epoch=0-step=755.ckpt',
#     'local_model_id': '/home/cezar/Projects/irmas_torch/outputs/2022-08-15/01-37-08/default/checkpoints/epoch=0-step=755.ckpt',
#     'framework': 'PyTorch',
#     'task': <clearml.task.Task object at 0x7f878e9b2860>}
#     """
#     # type(str, WeightsFileHandler.ModelInfo) -> Optional[WeightsFileHandler.ModelInfo]
#     assert operation_type in ("load", "save")
#     print(operation_type, model_info.__dict__)
#     # return None means skip the file upload/log, returning model_info will continue with the log/upload
#     # you can also change the upload destination file name model_info.upload_filename or check the local file size with Path(model_info.local_model_path).stat().st_size
#     return None


@hydra.main(config_path="config", config_name="config", version_base="1.1")
def main(cfg: DictConfig):

    data_module = lm.DataModule(**cfg.datamodule)
    print(cfg.datamodule)
    model = lm.ResnetIrmas(**cfg)
    logger = pl.loggers.TensorBoardLogger(".", name=cfg.experiment_name)

    callbacks = [
        pl.callbacks.early_stopping.EarlyStopping(
            monitor=cfg.watch_metric,
            min_delta=0.00,
            patience=cfg.early_stopping_patience,
            verbose=False,
            mode=watch_metric_mode,
        ),
        pl.callbacks.ModelCheckpoint(
            dirpath=f"{logger.save_dir}/checkpoints",
            monitor=cfg.watch_metric,
            mode=cfg.watch_metric_mode,
            auto_insert_metric_name=True,
            save_last=True,
        ),
        pl.callbacks.LearningRateMonitor(),
    ]

    trainer = pl.Trainer(
        gpus=1,
        logger=logger,
        max_epochs=cfg.max_epochs,
        callbacks=callbacks,
    )
    trainer = hydra.utils.instantiate(
        cfg.trainer,
        logger=logger,
        callbacks=callbacks,
    )
    trainer.fit(model, train_dataloaders=data_module)


if __name__ == "__main__":

    main()
