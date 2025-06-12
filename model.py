import torch
from torch import nn
from torchsummary import summary

class LetNet(nn.Module):

    def __init__(self):
        """
        模型每一层的设定
        """
        super(LetNet, self).__init__()
        self.c1 = nn.Conv2d(in_channels=1, out_channels=6, kernel_size=5, padding=2)
        self.relu = nn.ReLU()
        self.s2 = nn.AvgPool2d(kernel_size=2, stride=2)
        self.c3 = nn.Conv2d(in_channels=6, out_channels=16, kernel_size=5)
        self.s4 = nn.AvgPool2d(kernel_size=2, stride=2)
        self.flatten = nn.Flatten()
        # 使用softmax前的输出[batch_size, 10(10个分类)]
        self.softmax = nn.Softmax(dim=1)
        # 全神经网络
        self.f5 = nn.Linear(in_features=5 * 5 * 16, out_features=120)
        self.f6 = nn.Linear(in_features=120, out_features=84)
        self.f7 = nn.Linear(in_features=84, out_features=10)

    def forward(self, x):
        """
        前向传播
        :param x: 输入的图片数据
        :return: 输出图片的分类
        """
        x = self.relu(self.c1(x))
        x = self.s2(x)
        x = self.relu(self.c3(x))
        x = self.s4(x)
        x = self.flatten(x)
        x = self.f5(x)
        x = self.relu(x)
        x = self.f6(x)
        x = self.relu(x)
        x = self.f7(x)
        # x = self.softmax(x)
        return x

if __name__ == "__main__":
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = LetNet().to(device)
    print(summary(model, (1, 28, 28)))