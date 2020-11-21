from networks import SvddNet
from base.base_trainer import BaseTrainer
from base.base_dataset import BaseADDataset
from base.base_net import BaseNet
from sklearn.metrics import matthews_corrcoef, roc_auc_score, accuracy_score, precision_recall_fscore_support, roc_curve
import torch.utils.data as Data

import logging
import time
import torch
import torch.optim as optim
import numpy as np

from optim import DeepSVDDTrainer

"""
Ajoy 使用异常数据训练SVDD网络
     输入：球心c；已经训练好（正常数据）的SVDD网络；所有的异常数据(dataset)
     输出：由异常数据训练得到的svdd网络结构
     问题：用对半径R进行调整吗？
"""
class Svdd_Anomaly_Trainer(BaseTrainer):

    def __init__(self, network: SvddNet, c, optimizer_name: str = 'adam', lr: float = 0.001, n_epochs: int = 150,
                 lr_milestones: tuple = (), batch_size: int = 128, weight_decay: float = 1e-6, device: str = 'cuda',
                 n_jobs_dataloader: int = 0):
        super().__init__(optimizer_name, lr, n_epochs, lr_milestones, batch_size, weight_decay, device,
                         n_jobs_dataloader)

        self.net = network
        self.c = c

        # join time
        self.test_auc = None
        self.test_time = None
        self.test_score = None
        self.test_f_score = None
        self.test_mcc = None
        self.test_ftr = None
        self.test_tpr = None
    """
    Ajoy
        输入：dataset（LSTM网络输出的异常数据的中间编码对应的数据集）
    """
    def train(self, dataset: BaseADDataset):
        logger = logging.getLogger()

        # Set device for networks
        net = self.net.to(self.device)

        # Ajoy 修改数据集的loader（直接分batch就好）
        train_anomaly_loader = Data.DataLoader(dataset=dataset,
                                               batch_size=self.batch_size,
                                               # 每次采样是否打乱顺序
                                               shuffle=False,
                                               # 子进程的数量
                                               # 如果子进程数大于0，说明要进行多线程编程(一定要有一个主函数)
                                               num_workers=self.n_jobs_dataloader)


        # Set optimizer (Adam optimizer for now)
        optimizer = optim.Adam(net.parameters(), lr=self.lr, weight_decay=self.weight_decay, amsgrad=self.optimizer_name == 'amsgrad')

        # Set learning rate scheduler
        scheduler = optim.lr_scheduler.MultiStepLR(optimizer, milestones=self.lr_milestones, gamma=0.1)

        # Training
        logger.info('Starting training with anomaly data...')
        start_time = time.time()
        net.train()
        for epoch in range(self.n_epochs):

            scheduler.step()
            if epoch in self.lr_milestones:
                logger.info('  LR scheduler: new learning rate is %g' % float(scheduler.get_lr()[0]))

            loss_epoch = 0.0
            n_batches = 0
            epoch_start_time = time.time()
            for data in train_anomaly_loader:
                inputs, _, _ = data
                inputs = inputs.to(self.device)

                # Zero the networks parameter gradients
                optimizer.zero_grad()

                # Update networks parameters via backpropagation: forward + backward + optimize
                outputs = net(inputs)

                # Ajoy c为一个list，所以需要转化为tensor类型
                c = torch.Tensor(self.c)
               # print("the shape of c:", c.shape)
               # print("the shape of output:", outputs.shape)

                dist = torch.sum((outputs - c) ** 2, dim=1)
                loss = - torch.mean(dist)
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

        logger.info('Finished training with anomaly data.')

        return net

    # Ajoy 输入的数据不对
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
        logger.info('Starting testing(after anomaly data trained)...')
        start_time = time.time()
        idx_label_score = []
        net.eval()
        with torch.no_grad():
            for data in test_loader:
                inputs, labels, idx = data
                inputs = inputs.to(self.device)
                outputs = net(inputs)

                c = torch.Tensor(self.c)

                dist = torch.sum((outputs - c) ** 2, dim=1)
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
        logger.info('Finished testing(after anomaly data trained).')



def find_optimal_cutoff(label, y_prob):
    """ 寻找最优阀值 - - 阿登指数  """
    fpr, tpr, thresholds = roc_curve(label, y_prob)
    y = tpr - fpr
    youden_index = np.argmax(y)
    optimal_threshold = thresholds[youden_index]
    point = [fpr[youden_index], tpr[youden_index]]
    return optimal_threshold, point

