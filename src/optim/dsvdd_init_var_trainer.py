from base.base_trainer import BaseTrainer
from base.base_dataset import BaseADDataset
from base.base_net import BaseNet
from torch.utils.data.dataloader import DataLoader
from sklearn.metrics import roc_auc_score, roc_curve, precision_recall_fscore_support

import logging
import time
import torch
import torch.optim as optim
import numpy as np
"""
Ajoy
    输入：正常数据的中间编码（8维）
    损失函数：方差
    输出：svdd网络结构（32维outputs）
    功能：在用AE参数对svdd网络进行初始化之后，再利用方差损失对模型进行训练，使数据更加聚集
         测试过程需要对球心进行初始化，即第一次运行结果的平均值（只对球心c进更新，不涉及半径和nu参数）
         加入初始化球心部分
         简单test()
"""
class DSVDDInitVarTrainer(BaseTrainer):

    def __init__(self, c, optimizer_name: str = 'adam', lr: float = 0.001, n_epochs: int = 150,
                 lr_milestones: tuple = (), batch_size: int = 128, weight_decay: float = 1e-6, device: str = 'cuda',
                 n_jobs_dataloader: int = 0):
        super().__init__(optimizer_name, lr, n_epochs, lr_milestones, batch_size, weight_decay, device,
                         n_jobs_dataloader)

        # Deep SVDD parameters
        self.c = torch.tensor(c, device=self.device) if c is not None else None

        # Results
        self.train_time = None
        self.test_time = None


    def train(self, dataset: BaseADDataset, dsvdd_init_net: BaseNet):
        print('c', self.c)
        logger = logging.getLogger()

        # Set device for networks
        dsvdd_init_net = dsvdd_init_net.to(self.device)

        # Get train data loader
        train_loader, _ = dataset.loaders(batch_size=self.batch_size, num_workers=self.n_jobs_dataloader)

        # Set optimizer (Adam optimizer for now)
        optimizer = optim.Adam(dsvdd_init_net.parameters(), lr=self.lr, weight_decay=self.weight_decay,
                               amsgrad=self.optimizer_name == 'amsgrad')

        # Set learning rate scheduler
        scheduler = optim.lr_scheduler.MultiStepLR(optimizer, milestones=self.lr_milestones, gamma=0.1)

        # Training
        logger.info('Starting training with var...')
        start_time = time.time()
        dsvdd_init_net.train()
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

                # Update networks parameters via backpropagation: forward + backward + optimize
                outputs = dsvdd_init_net(inputs)
                # AJoy 计算损失函数
                loss = var_loss(outputs)

                loss.backward()
                optimizer.step()

                loss_epoch += loss.item()
                n_batches += 1

            # log epoch statistics
            epoch_train_time = time.time() - epoch_start_time
            logger.info('  Epoch {}/{}\t Time: {:.3f}\t Loss: {:.8f}'
                        .format(epoch + 1, self.n_epochs, epoch_train_time, loss_epoch / n_batches))

        self.train_time = time.time() - start_time
        logger.info('Training time: %.3f' % self.train_time)
        logger.info('Finished training with var.')

        return dsvdd_init_net

    """
    Ajoy
        测试过程只将每个数据对应的方差损失输出
        因为直接对数据集进行加载，所以需要test_loader
    """
    def test(self, dataset: BaseADDataset, dsvdd_init_net: BaseNet, eps=0.1):
        logger = logging.getLogger()

        # Set device for networks
        dsvdd_init_net = dsvdd_init_net.to(self.device)

        # Get test data loader
        _, test_loader = dataset.loaders(batch_size=self.batch_size, num_workers=self.n_jobs_dataloader)

        # Testing
        logger.info('Starting testing with var...')
        start_time = time.time()
        loss_epoch = 0
        n_batches = 0
        dsvdd_init_net.eval()
        with torch.no_grad():
            for data in test_loader:
                inputs, labels, idx = data
                inputs = inputs.to(self.device)
                outputs = dsvdd_init_net(inputs)
                loss = var_loss(outputs)

                '''
                print(
                    # '\nidx:', idx,
                    '\nvar_loss:', scores)
                '''
                loss_epoch += loss.item()
                n_batches += 1
                # 每个batch的平均误差
            logger.info('Test set Loss: {:.8f}'.format(loss_epoch / n_batches))

        self.test_time = time.time() - start_time
        logger.info('Testing time: %.3f' % self.test_time)
        logger.info('Fininshed testing with var.')
    """
    AJoy
        利用方差损失初始化好的dsvdd网络对球心进行初始化
    """
    def init_center_c(self, dataset: BaseADDataset, dsvdd_init_net: BaseNet, eps=0.1):
        logger = logging.getLogger()
        # Set device for networks
        dsvdd_init_net = dsvdd_init_net.to(self.device)

        # Get test data loader
        _, test_loader = dataset.loaders(batch_size=self.batch_size, num_workers=self.n_jobs_dataloader)

        # Initialize hypersphere center c as the mean from an initial forward pass on the data.
        n_samples = 0
        c = torch.zeros(dsvdd_init_net.rep_dim, device=self.device)

        logger.info('Initializing center c...')
        dsvdd_init_net.eval()
        with torch.no_grad():
            for data in test_loader:
                # get the inputs of the batch
                inputs, _, _ = data
                inputs = inputs.to(self.device)
                # output为32维数据  batch * 32
                outputs = dsvdd_init_net(inputs)
                n_samples += outputs.shape[0]
                # 求每列的和
                c += torch.sum(outputs, dim=0)

        c /= n_samples

        # If c_i is too close to 0, set to +-eps. Reason: a zero unit can be trivially matched with zero weights.
        c[(abs(c) < eps) & (c < 0)] = -eps
        c[(abs(c) < eps) & (c > 0)] = eps

        logger.info('Center c initialized.')
        return c

# 定义损失函数
def var_loss(data):
    # 求数据列的平均值
    mean = torch.mean(data, dim=0, keepdim=True)
    data_z = data - mean
    # 矩阵的对应位相乘
    loss_z = torch.mul(data_z, data_z)
    # 求每行数据的和
    loss_row_z = torch.sum(loss_z, dim=1, keepdim=True)
    # 求损失函数
    loss = loss_row_z.mean()
    return loss

