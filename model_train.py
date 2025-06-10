# -*- coding: UTF-8 -*-
'''
@Project ：learn_dl_leNet_5 
@File    ：model_train.py.py
@Author  ：公众号：思维侣行
@Date    ：2025/6/10 12:09 
'''
import torch
from torch import nn
from torchvision.datasets import FashionMNIST
from torchvision import transforms
import torch.utils.data as Data
import numpy as np
import matplotlib.pyplot as plt
from model import LeNet

def train_val_data_process():
    train_data = FashionMNIST(root='./data',
                              train=True,
                              download=True,
                              transform=transforms.Compose([transforms.Resize(size=28), transforms.ToTensor()]))
    train_data, val_data = Data.random_split(train_data, [round(0.8 * len(train_data)), round(0.2 * len(train_data))])

    train_data_loader = Data.DataLoader(dataset=train_data,
                                        batch_size=128,
                                        shuffle=True,
                                        num_workers=8)

    val_data_loader = Data.DataLoader(dataset=val_data,
                                        batch_size=128,
                                        shuffle=True,
                                        num_workers=8)

    return train_data_loader, val_data_loader

def train_model_process(model, train_data_loader, val_data_loader, num_epochs):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    # 使用adam 优化器，学习率为0.001
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
    # 交叉熵损失函数
    criterion = nn.CrossEntropyLoss()
    # 将模型放入训练设备中
    model = model.to(device)



