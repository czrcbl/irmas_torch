# Environment and dependencies

Python **3.14**, managed with **uv** (see [decisions/001](decisions/001-uv-and-pep-621-instead-of-poetry.md)).

```sh
uv sync                                   # install into .venv
uv run python train.py                    # full training run (Hydra)
uv run python train.py --cfg job          # print the composed config only
uv run python train.py +trainer.fast_dev_run=5 \
    datamodule.dataloader.num_workers=2   # quick smoke test
```

`fast_dev_run` runs 5 train and 5 val batches with loggers disabled. It is the
quickest way to check that a dependency bump didn't break anything.

## Main stack and why each piece is there

| Package | Used for |
|---------|----------|
| `torch`, `torchvision` | networks; `torchvision.transforms.Compose` |
| `torchaudio` | `Resample` and `MelSpectrogram` transforms only |
| `soundfile` | reading WAV files ([decisions/002](decisions/002-load-audio-with-soundfile.md)) |
| `lightning` | training loop (`lightning.pytorch`, Lightning 2 API) |
| `torchmetrics` | multilabel recall/precision/F1 during validation |
| `hydra-core` | config composition from `config/` ([decisions/003](decisions/003-hydra-1-4-pre-release-for-python-3-14.md)) |
| `tensorboard` | backend required by Lightning's `TensorBoardLogger` |
| `scikit-learn` | `average_precision_score` for validation mAP |

## API changes that came with the 3.14 migration (2026-10-27)

The jump was from torch 1.12 / pytorch-lightning 1.7 / torchmetrics 0.x. Things
that had to change, in case old code or notebooks are revived:

- **Lightning 2** (`import lightning.pytorch as pl`):
  - `validation_epoch_end(outputs)` was removed. `ResnetIrmas` now appends
    each step's outputs to `self.val_outputs` and reduces them in
    `on_validation_epoch_end()`, then clears the list.
  - `Trainer(gpus=1)` was removed. Use `accelerator` and `devices`
    (`config/trainer/default.yaml`: `accelerator: auto`, `devices: 1`).
  - `TensorBoardLogger` no longer works without `tensorboard` installed.
- **torchmetrics ≥0.11**: metrics need an explicit task:
  `Recall(task="multilabel", num_labels=11, threshold=0.5)`.
- **torch ≥2.7**: `ReduceLROnPlateau(verbose=...)` is gone. The LR is
  visible through `LearningRateMonitor` instead.
- **Python 3.14**: Linux `multiprocessing` now defaults to `forkserver`
  instead of `fork`, so DataLoader workers receive the dataset by pickling.
  `IRMASDataset` and its transforms are picklable; keep it that way (no
  lambdas, open file handles or other unpicklable objects as attributes).

## Not migrated: legacy scripts

These were already broken before the migration. They reference
`datasets.IRMAS`, `irmas_torch.utils`, `MelSpecTransformRosa`, `ignite` and
`easydict`, none of which exist any more. They predate the Lightning + Hydra
rewrite:

- `test.py`, `benchmark.py`, `tests/unit_tests.py`, `irmas_torch/train.py`
- `nbs/Eval.ipynb` (not checked)
