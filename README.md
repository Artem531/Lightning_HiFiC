# Lightning_HiFiC
This is pytorch lightning wrapper of https://github.com/Justin-Tan/high-fidelity-generative-compression for multi-gpu and mixed precision training (GAN not tested yet)
## How to install
1) conda install pytorch torchvision torchaudio cudatoolkit=10.2 -c pytorch
2) pip install pytorch-lightning
3) install apex (python build) https://github.com/NVIDIA/apex
4) pip install opencv-python
5) pip install scikit-image
6) pip install autograd

## How to run?
python lightning.py

## How to edit Trainer parameters?
In lightning.py add new pl.Trainer parameters from https://pytorch-lightning.readthedocs.io/en/latest/common/trainer.html
