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
from model import LetNet
import pandas as pd

def train_val_data_process():
    """
    获取数据，并设定训练集和验证集
    :return: 返回训练集和验证集
    """
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
    """
    使用模型训练，并保存最佳的模型
    :param model: 模型
    :param train_data_loader: 训练集
    :param val_data_loader: 验证集
    :param num_epochs: 训练的轮次
    :return: 训练集和验证集的loss和acc
    """
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
        # 训练每个 batch
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

        # 验证每个 batch
        for step, (val_x, val_y) in enumerate(val_data_loader):
            # 把验证数据的特征和标签加入到系统中
            val_x = val_x.to(device)
            val_y = val_y.to(device)
            # 使用模型推理
            model.eval()
            # 使用验证数据推理的结果（正向传播）
            output_val = model(val_x)
            # 查找每一个行中最大值对应的行标
            pre_lab_val = torch.argmax(output_val, dim=1)
            # 计算loss值
            loss = criterion(output_val, val_y)
            # 对验证集的损失函数进行累加
            val_loss += loss.item() * val_x.size(0)
            # 如果预测正确，则准确度train_corrects加1
            val_corrects += torch.sum(pre_lab_val == val_y.data)
            # 当前用于训练的样本数量
            val_num += val_x.size(0)

        # 计算并保存每一轮的训练和验证集的Loss值和准确率
        # 训练集
        train_loss_all.append(train_loss / train_num)
        train_acc_all.append(train_corrects.double().item() / train_num)
        # 验证集
        val_loss_all.append(val_loss / val_num)
        val_acc_all.append(val_corrects.double().item() / val_num)

        print(f"{'-'*10} epoch : {epoch}/{num_epochs -1} {'-'*10}")
        print(f"train_loss: {train_loss_all[-1]}, train_acc: {train_acc_all[-1]}")
        print(f"val_loss: {val_loss_all[-1]}, val_acc: {val_acc_all[-1]}")

        if val_acc_all[-1] > best_acc:
            best_acc = val_acc_all[-1]
            best_model_wts = copy.deepcopy(model.state_dict())

        # 计算训练和验证的耗时
        time_cost = time.time() - since
        print(f"训练和验证耗费的时间{time_cost // 60:.0f}m {time_cost % 60:.0f}s ")
    # 选择最优的参数， 保存最优参数的模型
    torch.save(best_model_wts, "./model/best_model.pth")

    # 返回每一
    train_process = pd.DataFrame(data={"epoch":range(num_epochs),
                                       "train_loss_all":train_loss_all,
                                       "val_loss_all":val_loss_all,
                                       "train_acc_all":train_acc_all,
                                       "val_acc_all":val_acc_all,})
    return train_process

def matplot_acc_loss(tran_process):
    """
    使用训练集和验证集的loss值和精确度 acc来生成变化曲线
    :param tran_process:
    :return:
    """
    # 设置中文字体和解决负号显示问题
    plt.rcParams['font.sans-serif'] = ['WenQuanYi Micro Hei'] #linux安装的字体：文泉驿微米黑
    plt.rcParams['axes.unicode_minus'] = False  # 解决负号 '-' 显示为方块的问题

    # 图形的宽度为 12 英寸，高度为 4 英寸。
    plt.figure(figsize=(12, 4))
    # 第一个图
    plt.subplot(1, 2, 1)
    plt.plot(tran_process["epoch"], tran_process['train_loss_all'], "ro-",  label='train_loss')
    plt.plot(tran_process["epoch"], tran_process['val_loss_all'], "bs-", label='val_loss')
    plt.legend()
    plt.xlabel('epoch')
    plt.ylabel('loss')
    plt.title("损失变化曲线")
    plt.grid(True)

    # 第二个图
    plt.subplot(1, 2, 2)
    plt.plot(tran_process["epoch"], tran_process['train_acc_all'], "ro-",  label='train_acc')
    plt.plot(tran_process["epoch"], tran_process['val_acc_all'], "bs-", label='val_acc')
    plt.legend()
    plt.xlabel('epoch')
    plt.ylabel('acc')
    plt.title("精确度变化曲线")
    plt.grid(True)

    plt.show()

if __name__ == '__main__':
    # 实例化模型
    LetNet = LetNet()
    # 加载数据集
    train_data, val_data = train_val_data_process()
    # 训练
    train_process = train_model_process(LetNet, train_data, val_data, 20)
    # 生成变化曲线图
    matplot_acc_loss(train_process)



