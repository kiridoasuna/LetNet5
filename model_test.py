# -*- coding: UTF-8 -*-
'''
@Project ：LetNet5 
@File    ：model_test.py
@Author  ：公众号：思维侣行
@Date    ：2025/6/13 02:22 
'''

import torch
import torch.utils.data as Data
from torchvision.datasets import FashionMNIST
from torchvision import transforms
from model import LetNet

def test_data_loader_process():
    test_data = FashionMNIST(root='./data', train=False, transform=transforms.Compose([transforms.ToTensor()]))
    test_data_loader = Data.DataLoader(test_data, batch_size=1, shuffle=False, num_workers=0)

    return test_data_loader

def test_model_process(model, test_data_loader):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    model.to(device)

    test_correct = 0.0
    total_test_num = len(test_data_loader.dataset)

    with torch.no_grad():
        for step, (test_data_x, test_data_y) in enumerate(test_data_loader):
            # 将特征放入到测试设备中
            test_data_x = test_data_x.to(device)
            # 将标签放入到测试设备中
            test_data_y = test_data_y.to(device)
            model.eval()

            output = model(test_data_x)

            pre_lab = torch.argmax(output, dim=1)
            test_correct += torch.sum(pre_lab == test_data_y).item()

        test_acc = test_correct / total_test_num

        print(f"Test Accuracy: {test_acc}")

if __name__ == '__main__':
    model = LetNet()
    model.load_state_dict(torch.load("./model/best_model.pth"))

    test_data = test_data_loader_process()
    test_model_process(model, test_data)




