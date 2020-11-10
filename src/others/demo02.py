import torch.optim as optim
import torch.nn as nn
import torch.nn.functional as F
import torch

'''
    torch 教程 -- 神经网络（LeNet为例）
        1. 定义包含可学习参数（权重）的神经网络模型
        2. 在数据集迭代
        3. 通过神经网络处理输入
        4. 计算损失
        5. 将梯度反向传播会网络节点
        6. 更新网路参数，梯度下降
    
    LeNet网络：
        1. 输入：   [1, 1, 32, 32]         
        2. 卷积层   [1, 6, 30, 30]
        3. 池化层   [1, 6, 15, 15]
        4. 卷积层   [1, 16, 13, 13]
        5. 池化层   [1, 16, 6, 6]
        6. 折叠层   [1, 576]
        7. 线性层   [1, 120]
        8. 线性层   [1, 84]
        9. 线性层   [1, 10]

    常见函数汇总
        view ：数据不变，改变数据维度，-1代表未知维度，系统自动生成

'''

class LeNet(nn.Module):
    def __init__(self):
        super(LeNet, self).__init__()

        # 卷积核
        self.conv1 = nn.Conv2d(1, 6, 3)  # 输入通道数 1 ，输出通道数 6，卷积核维度 3*3
        self.conv2 = nn.Conv2d(6, 16, 3)  # 输入通道数 6 ，输出通道数 16，卷积核维度 3*3

        # 线性运算
        self.fc1 = nn.Linear(16 * 6 * 6, 120)
        self.fc2 = nn.Linear(120, 84)
        self.fc3 = nn.Linear(84, 10)


    def forward(self, x):
        # (2, 2)窗口的最大池
        print(x.size())  # torch.Size([1, 1, 32, 32])
        x = F.relu(self.conv1(x))  # torch.Size([1, 6, 30, 30])
        print(x.size())
        x = F.max_pool2d(x, (2, 2))  # torch.Size([1, 6, 15, 15])
        print(x.size())
        x = F.relu(self.conv2(x))  # torch.Size([1, 16, 13, 13])
        print(x.size())
        x = F.max_pool2d(x, 2)  # torch.Size([1, 16, 6, 6])
        print(x.size())
        x = x.view(-1, self.num_flat_features(x))  # torch.Size([1, 576])
        print(x.size())
        x = F.relu(self.fc1(x))  # torch.Size([1, 120])
        print(x.size())
        x = F.relu(self.fc2(x))  # torch.Size([1, 84])
        print(x.size())
        x = self.fc3(x) # torch.Size([1, 10])
        print(x.size())
        return x

    def num_flat_features(self, x):
        size = x.size()[1:]
        num_features = 1
        for s in size:
            num_features *= s
        return num_features

print("###################  torch 教程 -- 神经网络  ####################")
net =LeNet()
print(net)

# 查看网络的参数
params = list(net.parameters())
print(len(params))
# print(params)
print("conv1's .weight", params[0].size())  # conv1's .weight

# 定义输入数据，并输出
input = torch.randn(1, 1, 32, 32)
out = net(input)
print(out)

# 将参数梯度清零，然后进行随机梯度的反向传播
net.zero_grad()
out.backward(torch.randn(1, 10))

# 损失函数
output = net(input)  # 输出值
target = torch.randn(10)  # 目标值
target = target.view(1, -1)
criterion = nn.MSELoss()

loss = criterion(output, target)
print(loss)
print(loss.grad_fn)  # MSELoss
print(loss.grad_fn.next_functions[0][0])  # Linear
print(loss.grad_fn.next_functions[0][0].next_functions[0][0])  # ReLU


# 反向传播
net.zero_grad()  # 清除梯度
print('conv1.bias.grad before backward')
print(net.conv1.bias.grad)

loss.backward()

print('conv1.bias.grad after backward')
print(net.conv1.bias.grad)

# 更新权重 w = w - learning rate * gradient
learning_rate = 0.01
for f in net.parameters():
    f.data.sub_(f.grad.data * learning_rate)

# 优化模型
optimizer = optim.SGD(net.parameters(), lr=0.01)

# 下面代码一般要执行多次，来优化模型
optimizer.zero_grad()
output = net(input)
loss = criterion(output, target)
loss.backward()
optimizer.step()
print(loss)
