# -*- coding: UTF-8 -*-
'''
@Project ：learn_dl_leNet_5 
@File    ：model_train.py.py
@Author  ：公众号：思维侣行
@Date    ：2025/6/10 12:09 
'''
import copy
import time

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
    # nn.CrossEntropyLoss是一个的复合函数，先是执行softmax,后执行负对数似然损失（NLL Loss）
    criterion = nn.CrossEntropyLoss()
    # 将模型放入训练设备中
    model = model.to(device)
    # 复制当前模型的参数组合函数，自带
    best_model_wts = copy.deepcopy(model.state_dict())

    # 初始化参数
    # 最高精确度
    best_acc = 0.0
    # 训练集损失值的列表
    train_loss_all = []
    # 验证集损失值的列表
    val_loss_all = []
    # 训练集的精度列表
    train_acc_all = []
    # 验证集损失函数列表
    val_acc_all = []
    # 当前时间
    since = time.time()

    for epoch in range(num_epochs):
        print(f"Epoch {epoch}/{num_epochs -1 }")
        print("-" * 10)

        # 初始化参数
        # 训练集损失值
        train_loss = 0.0
        # 训练集精确度
        train_corrects = 0
        # 验证集损失函数
        val_loss = 0.0
        # 验证集精确度
        val_corrects = 0
        # 训练集样本数量
        train_num = 0
        # 验证集样本数量
        val_num = 0

        for step, (b_x, b_y) in enumerate(train_data_loader):
            # 将数据-特征传入到设备中
            b_x = b_x.to(device)
            # 将数据-标签传入到设备中
            b_y = b_y.to(device)
            # 将模型设置为训练模型
            model.train()
            # 前向传播 出入一个batch的数据，输出一个batch中对应的预测
            output = model(b_x)
            # 查找每一个行中最大值对应的行标
            pre_lab = torch.argmax(output, dim=1)
            # 计算损失值
            loss = criterion(output, b_y)
            # 将梯度初始化为0
            optimizer.zero_grad()
            # 反向传播
            loss.backward()
            # 参数更新
            optimizer.step()

            # 对损失函数进行累加
            train_loss += loss.item() * b_x.size(0)
            # 如果预测正确，则准确度train_corrects加1
            train_corrects += torch.sum(pre_lab == b_y.data)
            # 当前用于训练的样本数量
            train_num += b_x.size(0)



