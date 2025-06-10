# -*- coding: UTF-8 -*-
"""
@Project ：learn_dl_leNet_5 
@File    ：plot.py.py
@Author  ：公众号：思维侣行
@Date    ：2025/6/6 23:02 
"""

from torchvision.datasets import FashionMNIST
from torchvision import transforms
import torch.utils.data as Data
import numpy as np

train_data = FashionMNIST(root='./data',
                          train=True,
                          download=True,
                          transform=transforms.Compose([transforms.Resize(size=224), transforms.ToTensor()]))

train_loader = Data.DataLoader(dataset=train_data,
                               batch_size=64,
                               shuffle=True,
                               num_workers=0)