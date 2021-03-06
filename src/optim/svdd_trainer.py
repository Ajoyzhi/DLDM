from base.base_trainer import BaseTrainer
from base.base_dataset import BaseADDataset
from base.base_net import BaseNet
from torch.utils.data.dataloader import DataLoader
from sklearn.metrics import matthews_corrcoef, roc_auc_score, accuracy_score, precision_recall_fscore_support, roc_curve


import logging
import time
import torch
import torch.optim as optim
import numpy as np

"""
Ajoy DSVDD中整个计算过程：
     1. 利用初始化后的DSVDD网络初始化球心c（之后球心c就不变了）init_center_c
     2.1 利用软边界函数更新参数W和R（利用参数nu更新）
     2.2 oc-svdd损失更新参数W
     【重点还要看怎么应用soft和oc损失函数】
"""
class SvddTrainer(BaseTrainer):
    """
        关键属性
            c：球心坐标，（32, ）第一次训练结果的平均值
            R：球面半径，（128,）
            dist：输出到球心的距离，（128,）
            output：每个输出32维，（128, 32）

        半径 R 如何获取？
            策略一：在测试时，把每一个训练集代入，得到一个输出，得到一个dist集合，找到最小的dist集合。
            策略二：与 soft-boundary 采用相同策略。
            np.quantile(np.sqrt(dist.clone().data.cpu().numpy()), 1 - nu)

    """

    def __init__(self, objective, R, c, nu: float, optimizer_name: str = 'adam', lr: float = 0.001, n_epochs: int = 150,
                 lr_milestones: tuple = (), batch_size: int = 128, weight_decay: float = 1e-6, device: str = 'cuda',
                 n_jobs_dataloader: int = 0):
        super().__init__(optimizer_name, lr, n_epochs, lr_milestones, batch_size, weight_decay, device,
                         n_jobs_dataloader)

        assert objective in ('one-class', 'soft-boundary'), "Objective must be either 'one-class' or 'soft-boundary'."
        self.objective = objective

        # Deep SVDD parameters
        self.R = torch.tensor(R, device=self.device)  # radius R initialized with 0 by default.
        self.c = torch.tensor(c, device=self.device) if c is not None else None
        # Ajoy 软件界中的系数mu
        self.nu = nu

        # Optimization parameters
        # Ajoy 半径R更新的频率
        self.warm_up_n_epochs = 10  # number of training epochs for soft-boundary Deep SVDD before radius R gets updated

        # join time
        self.test_auc = None
        self.test_time = None
        self.test_score = None
        self.test_f_score = None
        self.test_mcc = None
        self.test_ftr = None
        self.test_tpr = None

    def train(self, dataset: BaseADDataset, net: BaseNet):
        logger = logging.getLogger()

        # Set device for networks
        net = net.to(self.device)

        # Get train data loader
        train_loader, _ = dataset.loaders(batch_size=self.batch_size, num_workers=self.n_jobs_dataloader)

        # Set optimizer (Adam optimizer for now)
        optimizer = optim.Adam(net.parameters(), lr=self.lr, weight_decay=self.weight_decay,
                               amsgrad=self.optimizer_name == 'amsgrad')

        # Set learning rate scheduler
        scheduler = optim.lr_scheduler.MultiStepLR(optimizer, milestones=self.lr_milestones, gamma=0.1)

        # Initialize hypersphere center c (if c not loaded) Ajoy 函数在后面
        if self.c is None:
            logger.info('Initializing center c...')
            self.c = self.init_center_c(train_loader, net)
            logger.info('Center c initialized.')

        # Training
        logger.info('Starting training...')
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

                # Update networks parameters via backpropagation: forward + backward + optimize
                outputs = net(inputs)
                dist = torch.sum((outputs - self.c) ** 2, dim=1)

                if self.objective == 'soft-boundary':
                    scores = dist - self.R ** 2
                    loss = self.R ** 2 + (1 / self.nu) * torch.mean(torch.max(torch.zeros_like(scores), scores))
                else:
                    loss = torch.mean(dist)
                loss.backward()
                optimizer.step()

                # Update hypersphere radius R on mini-batch distances
                if (self.objective == 'soft-boundary') and (epoch >= self.warm_up_n_epochs):
                    # Ajoy 每10次迭代更新一次半径R，利用参数nu更新半径R
                    self.R.data = torch.tensor(get_radius(dist, self.nu), device=self.device)

                loss_epoch += loss.item()
                n_batches += 1

            # log epoch statistics
            epoch_train_time = time.time() - epoch_start_time
            logger.info('  Epoch {}/{}\t Time: {:.3f}\t Loss: {:.8f}'
                        .format(epoch + 1, self.n_epochs, epoch_train_time, loss_epoch / n_batches))

        self.train_time = time.time() - start_time
        logger.info('Training time: %.3f' % self.train_time)

        logger.info('Finished training.')

        return net

    def test(self, dataset: BaseADDataset, net: BaseNet, flag=0):
        logger = logging.getLogger()

        # Set device for networks
        net = net.to(self.device)

        # Get test data loader
        if flag == 0:
            _, test_loader = dataset.loaders(batch_size=self.batch_size, num_workers=self.n_jobs_dataloader)
        else:
            test_loader, _ = dataset.loaders(batch_size=self.batch_size, num_workers=self.n_jobs_dataloader)

        # Testing
        logger.info('Starting testing...')
        start_time = time.time()
        idx_label_score = []
        net.eval()
        with torch.no_grad():
            for data in test_loader:
                inputs, labels, idx = data
                inputs = inputs.to(self.device)
                outputs = net(inputs)
                dist = torch.sum((outputs - self.c) ** 2, dim=1)
                if self.objective == 'soft-boundary':
                    scores = dist - self.R ** 2
                else:
                    scores = dist

                # Save triples of (idx, label, score) in a list
                idx_label_score += list(zip(idx.cpu().data.numpy().tolist(),
                                            labels.cpu().data.numpy().tolist(),
                                            scores.cpu().data.numpy().tolist()))

        self.test_time = time.time() - start_time
        logger.info('Testing time: %.3f' % self.test_time)

        self.test_scores = idx_label_score

        # Compute AUC
        _, labels, scores = zip(*idx_label_score)
        labels = np.array(labels)
        scores = np.array(scores)

        self.test_score = scores
        self.test_auc = roc_auc_score(labels, scores)
        self.test_ftr, self.test_tpr, _ = roc_curve(labels, scores)
        optimal_threshold, _ = find_optimal_cutoff(labels, scores)
        pred_labels = np.array(list(map(lambda x: 0.0 if x <= optimal_threshold else 1.0, scores)))

        aucc = accuracy_score(labels, pred_labels)
        print("svdd_auc", aucc)

        self.test_mcc = matthews_corrcoef(labels, pred_labels)
        _, _, f_score, _ = precision_recall_fscore_support(labels, pred_labels, labels=[0, 1])
        self.test_f_score = f_score[1]

        logger.info('Test set AUC: {:.2f}%'.format(100. * self.test_auc))
        logger.info('Finished testing.')




    def init_center_c(self, train_loader: DataLoader, net: BaseNet, eps=0.1):
        """Initialize hypersphere center c as the mean from an initial forward pass on the data."""
        n_samples = 0
        c = torch.zeros(net.rep_dim, device=self.device)
        #Ajoy 因为c的初始化只是将所有的训练数据输入到SVDD中运行一次得到的output的平均值
        net.eval()
        with torch.no_grad():
            for data in train_loader:
                # get the inputs of the batch
                inputs, _, _ = data
                inputs = inputs.to(self.device)
                #Ajoy 一个batch的output
                outputs = net(inputs)
                #Ajoy 统计batch中的数据量
                n_samples += outputs.shape[0]
                #Ajoy 计算一个batch的output和
                c += torch.sum(outputs, dim=0)

        c /= n_samples

        # If c_i is too close to 0, set to +-eps. Reason: a zero unit can be trivially matched with zero weights.
        c[(abs(c) < eps) & (c < 0)] = -eps
        c[(abs(c) < eps) & (c > 0)] = eps

        return c


def find_optimal_cutoff(label, y_prob):
    """ 寻找最优阀值 - - 阿登指数  """
    fpr, tpr, thresholds = roc_curve(label, y_prob)
    y = tpr - fpr
    youden_index = np.argmax(y)
    optimal_threshold = thresholds[youden_index]
    point = [fpr[youden_index], tpr[youden_index]]
    return optimal_threshold, point


def get_radius(dist: torch.Tensor, nu: float):
    """Optimally solve for radius R via the (1-nu)-quantile of distances."""
    return np.quantile(np.sqrt(dist.clone().data.cpu().numpy()), 1 - nu)
