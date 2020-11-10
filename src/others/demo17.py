import torch
import numpy as np
import matplotlib.pyplot as plt

from torch import nn

""" 航班预测 lstm + 序列 """


class PlaneNet(nn.Module):
    def __init__(self):
        super(PlaneNet, self).__init__()
        """
        输入格式是1， 输出隐藏层大小为32
        对于小数据及num_layers，设置的很小
        num_layers就是lstm的层数，例如设置为2 ，就相当于有两个连续的lstm层
        原来输入格式为：(seq, batch, shape)
        设置batch_first = True 后，输入格式就可以改为：(batch，seq，shape)
        """
        self.lstm = nn.LSTM(input_size=1, hidden_size=32, num_layers=1, batch_first=True)
        self.linear = nn.Linear(32*seq, 1)

    def forward(self, x):
        x, (h, c) = self.lstm(x)
        x = x.reshape(-1, 32*seq)
        x = self.linear(x)
        return x


device = torch.device("cpu")

with open("F:/pycharm_workspace/lstm-svdd-master/data/lstm_plane/plane.csv", "r", encoding="utf-8") as f:
    data = f.read()
data = [row.split(',') for row in data.split("\n")]
value = [int(each[1]) for each in data]

li_x = []
li_y = []
seq = 2  # 序列长度为 2

for i in range(len(data) - seq):
    li_x.append(value[i: i+seq])  # 输入就是[x， x+1]天的航班数，输出就是x+2天的航班数
    li_y.append(value[i+seq])

train_x = (torch.tensor(li_x[:-30]).float() / 1000).reshape(-1, seq, 1).to(device)
train_y = (torch.tensor(li_y[:-30]).float() / 1000).reshape(-1, 1).to(device)

test_x = (torch.tensor(li_x[-30:]).float() / 1000.).reshape(-1, seq, 1).to(device)
test_y = (torch.tensor(li_y[-30:]).float() / 1000.).reshape(-1, 1).to(device)

model = PlaneNet().to(device)
optimizer = torch.optim.Adam(model.parameters(), lr=0.003)
loss_fun = nn.MSELoss()

model.train()
for epoch in range(300):
    output = model(train_x)
    loss = loss_fun(output, train_y)
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()
    if epoch % 20 == 0 and epoch > 0:
        test_loss = loss_fun(model(test_x), test_y)
        print("epoch:{}, loss:{}, test_loss: {}".format(epoch, loss, test_loss))

model.eval()
result = li_x[0][:seq-1] + list((model(train_x).data.reshape(-1))*1000) + list((model(test_x).data.reshape(-1))*1000)
# 通过模型计算预测结果并解码后保存到列表里，因为预测是从第seq个开始的，所有前面要加seq-1条数据
plt.plot(value, label="real")
# 原来的走势
plt.plot(result, label="pred")
# 模型预测的走势
plt.legend(loc='best')
plt.show()






