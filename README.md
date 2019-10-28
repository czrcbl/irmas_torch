# IRMAS TORCH

Convolutional Neural Networks applied to the [IRMAS](https://www.upf.edu/web/mtg/irmas).

This project applies known architectures to the task of predominant musical instrument identification in polyphonic recordings.

The networks are implemented in [PyTorch](https://pytorch.org/), and rely on [TorchAudio](https://pytorch.org/audio/) and [TorchVision](https://pytorch.org/docs/stable/torchvision/index.html).

## Content:
* The `src` folder contains all the utilities, such as transforms and datasets.
* The `train.py` file is a commandline utility to start the training process.
* The `test.py` file is a script to perform predictions on the the test set.
* On the `nbs` folder, notebooks displaying the results can be seen.
* The dataset, or a symlink to it, should be placed on the folder `data`.
* The trained models and predictions results are written on the `data` folder by default, this can be changed on the `config.py` file.
