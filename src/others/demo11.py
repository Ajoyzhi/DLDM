import torch
from torch import nn, optim
from torch.autograd import Variable
from torch.utils.data import DataLoader
from torchvision import datasets, transforms

'''  
[mnist分类](https://blog.csdn.net/qq_23144435/article/details/88692838)
lstm

'''

# 超参数
batch_size = 100
learning_rate = 0.01
num_epoches = 20

# 预处理
data_tf = transforms.Compose([transforms.ToTensor(), transforms.Normalize([0.5], [0.5])])
train_dataset = datasets.MNIST(root='F:/pycharm_workspace/svdd-pytorch-kdd99/data/lstm_mnist', train=True,
                               transform=data_tf, download=False)

test_dataset = datasets.MNIST(root='F:/pycharm_workspace/svdd-pytorch-kdd99/data/lstm_mnist', train=False,
                              transform=data_tf, download=False)
train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)


'''
建立RNN模型 -- 两层神经网络
    第一层为LSTM，用于进行序列分析
    第二层为全连接层，用于分类
    
参数说明
    input_size：28
    hidden_size ：128
    batch_size：100
    n_layer：2

默认维度
    输入层：[100, 28, 28]
    隐藏层：[100, 28, 128]
    变换层：[100, 128]
    输出层：[100, 10]
'''


class Rnn(nn.Module):
    def __init__(self, in_dim, hidden_dim, n_layer, n_class):
        super(Rnn, self).__init__()
        self.n_layer = n_layer
        self.hidden_dim = hidden_dim
        self.lstm = nn.LSTM(in_dim, hidden_dim, n_layer, batch_first=True)  # 输入层，隐藏层，层数
        self.n_layer = n_layer
        self.classifier = nn.Linear(hidden_dim, n_class)

    def forward(self, x):
        print(x.shape)
        out, _ = self.lstm(x)
        print(out.shape)  # [100, 28, 128]
        out = out[:, -1, :]
        print(out.shape)  # [100, 128]
        out = self.classifier(out)
        print(out.shape)  # [100, 10]
        return out


model = Rnn(28, 128, 2, 10)  # 图片大小是28x28
use_gpu = torch.cuda.is_available()
if use_gpu:
    model = model.cuda()

# 定义损失函数和优化器
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=learning_rate)

# 训练数据，评估性能
for epoch in range(num_epoches):
    running_loss = 0.0
    running_acc = 0.0
    for i, data in enumerate(train_loader, 1):
        img, label = data
        img = img.squeeze(1)
        if torch.cuda.is_available():
            img = img.cuda()
            label = label.cuda()
        else:
            img = Variable(img)
            label = Variable(label)

        # 前向传播 -- 计算损失
        out = model(img)
        loss = criterion(out, label)
        running_loss += loss.item() * label.size(0)
        _, pred = torch.max(out, 1)
        num_correct = (pred == label).sum()
        running_acc += num_correct.item()

        # 反向传播
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        print('[{}/{}] Loss: {:.6f}, Acc: {:.6f}'.format(
            epoch + 1, num_epoches, running_loss / (batch_size * i),
            running_acc / (batch_size * i)))

    print('Finish {} epoch, Loss: {:.6f}, Acc: {:.6f}'.format(
        epoch + 1, running_loss / (len(train_dataset)), running_acc / (len(
            train_dataset))))

model.eval()
eval_loss = 0.
eval_acc = 0.
for data in test_loader:
    img, label = data
    b, c, h, w = img.size()
    assert c == 1, 'channel must be 1'
    img = img.squeeze(1)
    # img = img.view(b*h, w)
    # img = torch.transpose(img, 1, 0)
    # img = img.contiguous().view(w, b, h)
    if use_gpu:
        img = Variable(img).cuda()
        label = Variable(label).cuda()
    else:
        img = Variable(img)
        label = Variable(label)
    out = model(img)
    loss = criterion(out, label)
    eval_loss += loss.item() * label.size(0)
    _, pred = torch.max(out, 1)
    num_correct = (pred == label).sum()
    eval_acc += num_correct.item()
print('Test Loss: {:.6f}, Acc: {:.6f}'.format(eval_loss / (len(
    test_dataset)), eval_acc / (len(test_dataset))))
