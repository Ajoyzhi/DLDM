import torch
import torch.nn as nn
import torch.nn.functional as F
from base.base_net import BaseNet


class KddNet(BaseNet):
    """ ae_kmeans, dsvdd使用该网络结构
        Ajoy 因为DLDM中通过LSTM将9维数据变为8维的中间编码，所以输入到DSVDD中为8维数据
             而对比实验，直接将9维数据输入，所以网络结构变化
     """
    def __init__(self):
        super().__init__()

        self.rep_dim = 32

        self.fc1 = nn.Linear(9,  16, bias=False)
        self.fc2 = nn.Linear(16, 32, bias=False)
        self.fc3 = nn.Linear(32, 64, bias=False)
        self.fc4 = nn.Linear(64, 32, bias=False)

    def forward(self, x):
        x = self.fc1(x)
        x = F.leaky_relu(x)
        x = self.fc2(x)
        x = F.leaky_relu(x)
        x = self.fc3(x)
        x = F.leaky_relu(x)
        x = self.fc4(x)
        return x


class KddNet_Autoencoder(BaseNet):

    def __init__(self):
        super().__init__()

        self.rep_dim = 32
        # encoder
        self.fe1 = nn.Linear(9, 16, bias=False)
        self.fe2 = nn.Linear(16, 32, bias=False)
        self.fe3 = nn.Linear(32, 64, bias=False)
        self.fe4 = nn.Linear(64, 32, bias=False)

        # decoder
        self.fd1 = nn.Linear(32, 64, bias=False)
        self.fd2 = nn.Linear(64, 32, bias=False)
        self.fd3 = nn.Linear(32, 16, bias=False)
        self.fd4 = nn.Linear(16, 9, bias=False)

    def forward(self, x):

        # encoder
        x = F.leaky_relu(self.fe1(x))
        x = F.leaky_relu(self.fe2(x))
        x = F.leaky_relu(self.fe3(x))
        x = self.fe4(x)

        # decoder
        x = self.fd1(x)
        x = F.leaky_relu(self.fd2(x))
        x = F.leaky_relu(self.fd3(x))
        x = torch.sigmoid(self.fd4(x))
        return x
