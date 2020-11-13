from base.base_dataset import BaseADDataset
from dsvdd_anomaly_trainer import Svdd_Anomaly_Trainer
from networks.main import build_network
from optim.lstm_trainer import LstmTrainer

"""
Ajoy
    train()函数中的数据集为训练集中异常数据通过LSTM网络的中间编码
"""
class DeepSVDDanomaly(object):
    def __init__(self, n_features=8):
        """初始化lstm参数."""
        self.n_features = n_features

        self.net_name = None
        self.net = None

        self.trainer = None
        self.optimizer_name = None

        self.results = {
            'train_time': None,
            'test_auc': None,
            'test_time': None,
            'test_scores': None,
        }

    def set_network(self, net_name):
        """Builds the neural networks ."""

        self.net_name = net_name
        self.net = build_network(net_name, self.n_features)

    def train(self, dataset: BaseADDataset, optimizer_name: str = 'RMSprop', lr: float = 0.001, n_epochs: int = 50,
              lr_milestones: tuple = (), batch_size: int = 128, weight_decay: float = 1e-6, device: str = 'cuda',
              n_jobs_dataloader: int = 0):
        """Trains the svdd on the training anomaly data."""

        self.optimizer_name = optimizer_name
        self.trainer = Svdd_Anomaly_Trainer(optimizer_name,
                                   lr=lr,
                                   n_epochs=n_epochs,
                                   lr_milestones=lr_milestones,
                                   batch_size=batch_size,
                                   weight_decay=weight_decay,
                                   device=device,
                                   n_jobs_dataloader=n_jobs_dataloader,
                                   n_features=self.n_features)
        # Get the model
        self.net = self.trainer.train(dataset, self.net)
        self.results['train_time'] = self.trainer.train_time

    def test(self, dataset: BaseADDataset, device: str = 'cuda', n_jobs_dataloader: int = 0):
        """Tests svdd model on the test data."""

        if self.trainer is None:
            self.trainer = Svdd_Anomaly_Trainer(device=device, n_jobs_dataloader=n_jobs_dataloader)

        self.trainer.test(dataset, self.net)  # 训练集

        # Get results
        self.results['test_auc'] = self.trainer.test_auc
        self.results['test_time'] = self.trainer.test_time
        self.results['test_scores'] = self.trainer.test_scores

