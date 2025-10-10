# AutoCalibrator-LFM

# System requirements
## Recommended Configuration

We recommend the following configurations:

* 32GB RAM or more
* NVIDIA GPU (supports CUDA) with more than 8GB VRAM

This repository is tested on a NVIDIA Geforce RTX 3080 (10GB RAM) and 256GB RAM.

## Environment
There is no restriction on operation systems or specific python versions (as long as they are supported by Pytorch). This repository is tested on Python 3.12.

## Installation
Clone this repository and install all dependencies as follow:
```shell
conda env create -f environment.yml
``````

It is worthing noting that the PyTorch installed by pip should be manually checked. The PyTorch package is expected to be supported by your CUDA installation in this repository. For more information about compabilities between PyTorch and CUDA, please check [INSTALLING PREVIOUS VERSIONS OF PYTORCH](https://pytorch.org/get-started/previous-versions/).

# Demo

We provide a demo, which contains a LF image. You can train the demo model by following steps:

1. Download the dataset ([here](https://drive.google.com/file/d/1ne0n3dMt27MbaHA13hNRaxn3guLNyVSl/view?usp=sharing)). Place all the TIFF file in `example` to `./data/image/raw`.
2. To start your training process, run
```bash
cd code
python train.py
```

# Correspondence
Should you have any questions regarding this code and the corresponding results, please contact Yuan Li(liyuan22@mails.tsinghua.edu.cn).
