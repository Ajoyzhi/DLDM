import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import matplotlib.pyplot as plt
import numpy as np
import torchvision
import torchvision.transforms as transforms

'''
    训练一个图像分类器
        1. 加载数据集
            一般来说加载图像，文本，音频，视频等数据时，可以使用标准的Python 来加载
            数据为Numpy数组，然后将这个数组转化为torch.tensor
                * 图像可以使用Pillow，OpenCV
                * 音频可以使用SciPy，librosa
                * 文本可以使用原始Python和Cython来加载，或者使用NLTK或SpaCy来处理
            特别的，对于图像加载，PyTorch提供了专门的包torchVision，它包含了一些基
            本图像数据集，这些数据集包括ImageNet，CIFAR10,MNIST等。除了加载数据以外
            ，torchVision还包含了图像转换器 torchvision.datasets 和 torch.utils.
            data.DataLoader 数据加载器。
            
            CAFAR10数据集：该数据集有10个类别，图像都是[3, 32, 32]
        
        2. 基本流程
            1. 使用 torchVision 加载和归一化 CIFAR10 训练集和测试集
            2. 定义一个卷积神经网络
            3. 定义损失函数
            4. 在训练集上训练网络
            5. 在测试集上测试网络
        
            
'''

'''
    读取和归一化 CIFAR10 
        使用 torchvision 可以非常容易地加载 CIFAR10.
        torchvision的输出是[0, 1]的PILImage图像
        我们把它转化为归一化范围为[-1, 1]的张量
'''


# 图像预处理步骤
transforms = transforms.Compose(
    [transforms.ToTensor(), transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))]
)


# 训练数据加载器
trainset = torchvision.datasets.CIFAR10(
    root='F:/pycharm_workspace/Deep-SVDD-PyTorch-master/dataTest', train=True,
    download=False, transform=transforms
)
print(len(trainset))
print(trainset.__getitem__(100))
trainloader = torch.utils.data.DataLoader(
    trainset, batch_size=4, shuffle=False, num_workers=0
)

# 测试数据加载器
testset = torchvision.datasets.CIFAR10(
    root="F:/pycharm_workspace/Deep-SVDD-PyTorch-master/dataTest", train=False,
    download=False, transform=transforms
)
testloader = torch.utils.data.DataLoader(
    testset, batch_size=4, shuffle=False, num_workers=0
)

# 图像类别
classes = ('plane', 'car', 'bird', 'cat', 'deer',
           'dog', 'frog', 'horse', 'ship', 'truck')

# 可视化函数
def imshow(img):
    img = img/2 + 0.5  # 反向归一化
    npimg = img.numpy()
    plt.imshow(np.transpose(npimg, (1, 2, 0)))
    plt.show()

# 获取随机数据
dataiter = iter(trainloader)
images, labels = dataiter.next()

# 展示图像
imshow(torchvision.utils.make_grid(images))

# 显示图像标签
print(' '.join('%5s' % classes[labels[j]] for j in range(4)))


# 定义一个神经网络
class CifarNet(nn.Module):
    def __init__(self):
        super(CifarNet, self).__init__()
        self.conv1 = nn.Conv2d(3, 6, 5)
        self.pool = nn.MaxPool2d(2, 2)
        self.conv2 = nn.Conv2d(6, 16, 5)
        self.fc1 = nn.Linear(16 * 5 * 5, 120)
        self.fc2 = nn.Linear(120, 84)
        self.fc3 = nn.Linear(84, 10)

    def forward(self, x):
        x = F.relu(self.conv1(x))
        x = self.pool(x)
        x = F.relu(self.conv2(x))
        x = self.pool(x)
        x = x.view(-1, 16 * 5 * 5)
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.fc3(x)
        return x

net = CifarNet()

# 定义损失函数和优化器
# 交叉熵作为损失函数，使用带动量的随机梯度下降完成参数优化
criterion = nn.CrossEntropyLoss()
optimizer = optim.SGD(net.parameters(), lr=0.001, momentum=0.9)

# 训练网络
for epoch in range(1):  # 迭代一次
    running_loss = 0.0
    for i, data in enumerate(trainloader, 0):
        inputs, labels = data
        optimizer.zero_grad()
        outputs = net(inputs)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()
        # 打印 log
        running_loss += loss.item()
        if i % 200 == 199:
            print('[%d, %5d] loss: %.3f' %
                  (epoch + 1, i + 1, running_loss / 200))
            running_loss = 0.0
print("Finished Training")


# 测试网络
dataiter = iter(testloader)
images, labels = dataiter.next()
imshow(torchvision.utils.make_grid(images))
print('GroundTruth: ', ' '.join('%5s' % classes[labels[j]] for j in range(4)))
outputs = net(images)
_, predicted = torch.max(outputs, 1)
print('Predicted: ', ' '.join('%5s' % classes[predicted[j]] for j in range(4)))

correct = 0
total = 0
with torch.no_grad():
    for data in testloader:
        images, labels = data
        outputs = net(images)
        _, predicted = torch.max(outputs.data, 1)
        total += labels.size(0)
        correct += (predicted == labels).sum().item()

print('Accuracy of the networks on the 10000 test images: %d%%' %
      (100 * correct / total))
