# PyTorch SimCLR: A Simple Framework for Contrastive Learning of Visual Representations

## Installation

```
$ conda env create --name hch --file env.yml
$ conda activate hch
$ python run.py
```

## Config file

Before running HCH for imagenet you have to download it manually.
Before running HCH, make sure you choose the correct running configurations. You can change the running configurations by passing keyword arguments to the ```run.py``` file.

```python

$ python run.py -dataset-name stl10 --log-every-n-steps 100 --epochs 100 --level_number 4

```

If you want to run it on CPU (for debugging purposes) use the ```--disable-cuda``` option.

For 16-bit precision GPU training, there **NO** need to to install [NVIDIA apex](https://github.com/NVIDIA/apex). Just use the ```--fp16_precision``` flag and this implementation will use [Pytorch built in AMP training](https://pytorch.org/docs/stable/notes/amp_examples.html).
