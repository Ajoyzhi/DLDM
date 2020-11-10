import torch
import torch.nn as nn
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import cross_val_score
from base_net import BaseNet

'''
[鸢尾花分类](https://blog.csdn.net/qq_23144435/article/details/88692838)
lstm + autoencoder

'''
# 超参数
EPOCH = 200
LR = 0.005

dataset = load_iris()
label = dataset.target  # autoencoder 不需要考虑标签
data = dataset.data

print(data.shape)
print(label.shape)
'''

维度变换
encoder
输入层：[150, 1, 4]
隐藏层：[150, 1, 64]
折叠层：[150, 64]
连接层：[150, 3]

decoder
输入层：[150, 1, 3]
隐藏层：[150, 1, 64]
折叠层：[150, 64]
连接层：[150, 4]

'''


class RNN(BaseNet):

    def __init__(self):
        super().__init__()
        self.lstm_encoder = nn.LSTM(
            input_size=4,
            hidden_size=64,
            num_layers=1,
            batch_first=True
        )
        self.fe = nn.Linear(in_features=64, out_features=3)

        self.lstm_decoder = nn.LSTM(
            input_size=3,
            hidden_size=64,
            num_layers=1,
            batch_first=True
        )
        self.fd = nn.Linear(in_features=64, out_features=4)

    def forward(self, x):
        x, _ = self.lstm_encoder(x)
        x = x[:, -1, :]
        x = self.fe(x)

        y = x.view(-1, 1, 3)

        y, _ = self.lstm_decoder(y)
        y = y[:, -1, :]
        y = self.fd(y)
        return x, y

def train(lstm_net: BaseNet):

    optimizer = torch.optim.Adam(lstm_net.parameters(), lr=0.001)
    loss_F = nn.MSELoss()
    for epoch in range(500):  # 数据集只迭代一次

        input = torch.from_numpy(data).unsqueeze(0).float()
        _, pred = lstm_net(input.view(-1, 1, 4))

        loss = loss_F(pred, input[0])  # 计算loss
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        print('Epoch: ', epoch, '| train loss: %.4f' % loss.data.numpy())
    return lstm_net

def test(lstm_net: BaseNet):
    pred, _ = lstm_net(input.view(-1, 1, 4))
    print("autoencoder' code is ", pred.shape)
    pred = pred.squeeze(1).detach().numpy()
    print(pred)

    knn = KNeighborsClassifier(n_neighbors=5)
    scores = cross_val_score(knn, pred, label, cv=6, scoring='accuracy')
    print(scores)

lstm_net = RNN()
lstm_net = train(lstm_net)
test(lstm_net)






