import torch
import torch.nn as nn

"""
联合训练，类似GAN训练
    把两个训练好的网络接到一起训练。
"""

x = torch.rand(2, 3)
net1 = nn.Linear(3, 3)
net2 = nn.Linear(3, 3)
a = net1(x)
b = net2(a)

tgt = torch.rand(2, 3)
loss_fun = torch.nn.MSELoss()
opt1 = torch.optim.Adam(net1.parameters(), 0.002)
opt2 = torch.optim.Adam(net2.parameters(), 0.002)


print(net1.bias)
print(net2.bias)

for i in range(100):
    tmp = net1(x)
    output = net2(tmp)
    loss = loss_fun(output, tgt)

    net1.zero_grad()
    net2.zero_grad()

    loss.backward()
    opt1.step()  # 更新参数
    # opt2.step()

    print('EPOCH:{},LOSS={}'.format(i, loss))

aa = net1(x)
bb = net1(aa)

print(net1.bias)
print(net2.bias)

print(a)
print(aa)
print(b)
print(bb)



