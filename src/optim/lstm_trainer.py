from base.base_trainer import BaseTrainer
from base.base_dataset import BaseADDataset
from base.base_net import BaseNet
from networks import LstmNet
from sklearn.metrics import roc_auc_score

import torch
import torch.optim as optim
import logging
import time
import numpy as np

"""
Ajoy:加入函数get_code()
     输入：数据集，LSTM网络
     输出：LSTM的中间编码及对应的label
     调用：lstm。get_code(dataset)
"""
class LstmTrainer(BaseTrainer):
    """
    lstm 具体实现
        训练 lstm 网络，输出 train_code 和 train_label.
        测试 lstm 网络，输出 test_code 和 test_label.

    属性
        train_code []:训练 lstm 时, 每条数据对应的码 .
        test_code []: 测试 lstm 时, 每条数据对应的码.
        train_label []: 测试 lstm 时, 每条数据对应的标签
        test_label []: 训练 lstm 时,每条数据对应的标签

        epsilon: 优化器参数
        momentum: 优化器参数
        l2_decay: 优化器参数

    """

    def __init__(self,
                 optimizer_name: str = 'adam',
                 lr: float = 0.001,
                 n_epochs: int = 150,
                 lr_milestones: tuple = (),
                 batch_size: int = 128,
                 weight_decay: float = 1e-6,
                 device: str = 'cuda',
                 n_jobs_dataloader: int = 0,
                 n_features=8):
        super().__init__(optimizer_name, lr, n_epochs, lr_milestones, batch_size, weight_decay, device,
                         n_jobs_dataloader)

        self.n_features = n_features
        self.epsilon = 0.0001
        self.momentum = 0.9
        self.l2_decay = 0.0001

        self.train_code = []
        self.train_label = []

        self.test_code = []
        self.test_label = []

        self.train_time = None
        self.test_auc = None
        self.test_time = None
        self.test_scores = None

    def train(self, dataset: BaseADDataset, net: BaseNet):
        logger = logging.getLogger()

        # Set device for networks
        net = net.to(self.device)

        # Ajoy 其实是TorchvisionDataset中得到loader方法，可以加载训练集和测试集
        train_loader, _ = dataset.loaders(batch_size=self.batch_size, num_workers=self.n_jobs_dataloader)

        optimizer = optim.RMSprop(net.parameters(), lr=self.lr, weight_decay=self.weight_decay, eps=self.epsilon,
                                  momentum=self.momentum)

        # Set learning rate scheduler
        scheduler = optim.lr_scheduler.MultiStepLR(optimizer, milestones=self.lr_milestones, gamma=0.1)

        # Training
        logger.info('Starting train lstm_autoencoder ...')
        start_time = time.time()
        net.train()
        for epoch in range(self.n_epochs):

            scheduler.step()
            if epoch in self.lr_milestones:
                logger.info('  LR scheduler: new learning rate is %g' % float(scheduler.get_lr()[0]))

            loss_epoch = 0.0
            n_batches = 0
            epoch_start_time = time.time()
            for data in train_loader:
                inputs, _, _ = data
                inputs = inputs.to(self.device)
                # Zero the networks parameter gradients
                optimizer.zero_grad()

                # Update networks parameters via back propagation: forward + backward + optimize
                _, outputs = net(inputs.view(-1, 1, self.n_features))
                #Ajoy 均方误差
                scores = torch.sum((outputs - inputs) ** 2, dim=tuple(range(1, outputs.dim())))
                loss = torch.mean(scores)
                loss.backward()
                optimizer.step()

                loss_epoch += loss.item()
                n_batches += 1

            # log epoch statistics
            epoch_train_time = time.time() - epoch_start_time
            logger.info('  Epoch {}/{}\t Time: {:.3f}\t Loss: {:.8f}'
                        .format(epoch + 1, self.n_epochs, epoch_train_time, loss_epoch / n_batches))

        self.train_time = time.time() - start_time
        logger.info('lstm_autoencoder train time: %.3f' % self.train_time)
        logger.info('Finished train lstm_autoencoder.')

        return net

    def test(self, dataset: BaseADDataset, net: BaseNet, is_test=0):
        """
            dt_type：数据集的类型， 测试集 0 / 训练集 1 /
        """
        logger = logging.getLogger()

        # Set device for networks
        net = net.to(self.device)

        # Get test data loader
        if is_test == 0:  # 测试集加载器
            _, test_loader = dataset.loaders(batch_size=self.batch_size, num_workers=self.n_jobs_dataloader)
        if is_test == 1:  # 训练集加载器
            test_loader, _ = dataset.loaders(batch_size=self.batch_size, num_workers=self.n_jobs_dataloader)

        # Testing
        logger.info('Testing lstm_autoencoder...')
        loss_epoch = 0.0
        n_batches = 0
        start_time = time.time()
        idx_label_score = []
        net.eval()
        with torch.no_grad():
            for data in test_loader:
                inputs, labels, idx = data
                inputs = inputs.to(self.device)

                # get lstm test label，label.shape = (128,)
                label = labels.numpy()

                if is_test == 0:
                    for i in range(len(label)):
                        self.test_label.append(label[i])
                if is_test == 1:
                    for i in range(len(label)):
                        self.train_label.append(label[i])

                # Ajoy LSTM还可以直接输出中间编码嘛？
                # Ajoy LSTMnet中定义的forward函数中返回了encoder和decoder两个函数的值
                code, outputs = net(inputs.view(-1, 1, self.n_features))
                # Ajoy 从当前图中返回一个新的tensor，该tensor不需要计算梯度
                code = code.detach().numpy()

                if is_test == 0:
                    for i in range(len(code)):
                        # Ajoy 将测试数据集的中间编码放入test_code数组
                        # 【为什么要用追加？因为一次只能对一个batch的数据进行操作，所以只能得到一个batch的code，所以每次要追加】
                        self.test_code.append(code[i])
                if is_test == 1:
                    for i in range(len(code)):
                        # Ajoy 将训练集的中间编码放入train_code数组中
                        self.train_code.append(code[i])
                # 均方误差
                scores = torch.sum((outputs - inputs) ** 2, dim=tuple(range(1, outputs.dim())))
                loss = torch.mean(scores)

                # Save triple of (idx, label, score) in a list
                idx_label_score += list(zip(idx.cpu().data.numpy().tolist(),
                                            labels.cpu().data.numpy().tolist(),
                                            scores.cpu().data.numpy().tolist()))

                loss_epoch += loss.item()
                n_batches += 1
        # 每个batch的平均误差
        logger.info('Test set Loss: {:.8f}'.format(loss_epoch / n_batches))

        self.test_time = time.time() - start_time
        logger.info('lstm_autoencoder testing time: %.3f' % self.test_time)
        self.test_scores = idx_label_score

        _, labels, scores = zip(*idx_label_score)
        labels = np.array(labels)
        scores = np.array(scores)

        print(len(labels))
        print(len(scores))

        """ 测试集 """
        if is_test == 0:
            self.test_auc = roc_auc_score(labels, scores)
            logger.info('Test set AUC: {:.2f}%'.format(100. * self.test_auc))
            logger.info('Finished test lstm_autoencoder.')

    def load_code(self):
        """ 加载 lstm 网络输出的 code 和 label """
        return self.train_code, self.train_label, self.test_code, self.test_label

    # 获取输入数据集的中间编码
    """
    AJoy
        输入 :dataset(kdd99数据集)，net（lstm网络）
        输出：dataset.anomaly_loaders（异常数据）的中间编码和标签
    """
    def get_code(self, dataset: BaseADDataset, net: LstmNet):
        # 定义输出的编码和标签
        other_code = []
        other_label = []

        logger = logging.getLogger()

        # Set device for networks
        net = net.to(self.device)

        # Get anomaly dataset_loader
        anomaly_loader = dataset.anomaly_loaders(batch_size=self.batch_size, num_workers=self.n_jobs_dataloader)

        # Testing
        logger.info('Getting the code of the dataset...')
        net.eval()
        with torch.no_grad():
            for data in anomaly_loader:
                inputs, labels, idx = data
                inputs = inputs.to(self.device)

                # get lstm test label，label.shape = (128,)
                label = labels.numpy()

                for i in range(len(label)):
                    other_label.append(label[i])

                # Ajoy LSTMnet中定义的forward函数中返回了encoder和decoder两个函数的值
                code, _ = net(inputs.view(-1, 1, self.n_features))
                # Ajoy 从当前图中返回一个新的tensor，该tensor不需要计算梯度
                code = code.detach().numpy()

                for i in range(len(code)):
                    # Ajoy 将测试数据集的中间编码放入test_code数组
                    # 【为什么要用追加？因为一次只能对一个batch的数据进行操作，所以只能得到一个batch的code，所以每次要追加】
                    other_code.append(code[i])

        logger.info('Finished getting code.')

        return other_code, other_label