import torch
import torch.nn as nn
from torch.autograd import Variable
import torch.distributed as dist
import torch.utils.data as data_utils


class Net(nn.Module):
    def __init__(self, n_input, n_hidden, n_output):
        super(Net, self).__init__()
        self.hidden1 = nn.Linear(n_input, n_hidden)
        self.hidden2 = nn.Linear(n_hidden, n_hidden)

        self.predict1 = nn.Linear(n_hidden * 2, n_output)
        self.predict2 = nn.Linear(n_hidden * 2, n_output)

    def forward(self, input1, input2):  # 多输入！！！
        out01 = self.hidden1(input1)
        out02 = torch.relu(out01)
        out03 = self.hidden2(out02)
        out04 = torch.sigmoid(out03)

        out11 = self.hidden1(input2)
        out12 = torch.relu(out11)
        out13 = self.hidden2(out12)
        out14 = torch.sigmoid(out13)

        out = torch.cat((out04, out14), dim=1)  # 模型层拼合！！！当然你的模型中可能不需要~

        out1 = self.predict1(out)
        out2 = self.predict2(out)

        return out1, out2  # 多输出！！！


net = Net(1, 20, 1)

x1 = torch.unsqueeze(torch.linspace(-1, 1, 100), dim=1)  # 请不要关心这里，随便弄一个数据，为了说明问题而已
y1 = x1.pow(3) + 0.1 * torch.randn(x1.size())

x2 = torch.unsqueeze(torch.linspace(-1, 1, 100), dim=1)
y2 = x2.pow(3) + 0.1 * torch.randn(x2.size())

x1, y1 = (Variable(x1), Variable(y1))
x2, y2 = (Variable(x2), Variable(y2))

optimizer = torch.optim.SGD(net.parameters(), lr=0.1)
loss_func = torch.nn.MSELoss()

for t in range(5000):
    prediction1, prediction2 = net(x1, x2)
    loss1 = loss_func(prediction1, y1)
    loss2 = loss_func(prediction2, y2)
    loss = loss1 + loss2  # 重点！

    optimizer.zero_grad()
    loss.backward()
    optimizer.step()

    if t % 100 == 0:
        print('Loss1 = %.4f' % loss1.data, 'Loss2 = %.4f' % loss2.data, )

