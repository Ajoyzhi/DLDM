import json
import torch

from base.base_dataset import BaseADDataset
from networks.main import build_network, build_autoencoder
from optim.deepSVDD_trainer import DeepSVDDTrainer
from optim.ae_trainer import AETrainer
from lstm import Lstm


class DeepSVDD(object):
    """
        dldm-svdd 部分的逻辑代码
        Ajoy DSVDD 相关参数设置
                   train() & test() & pretrain() & 利用预训练网络初始化 & 保存DSVDD网络和预训练网络 & 加载网络
    """

    def __init__(self, lstm: Lstm, objective: str = 'one-class', nu: float = 0.1, n_code=8):
        """Inits DeepSVDD with one of the two objectives and hyperparameter nu."""

        assert objective in ('one-class', 'soft-boundary'), "Objective must be either 'one-class' or 'soft-boundary'."
        self.objective = objective
        assert (0 < nu) & (nu <= 1), "For hyperparameter nu, it must hold: 0 < nu <= 1."
        self.nu = nu
        self.n_code = n_code
        self.R = 0.0
        self.c = None

        self.c_tensor = None
        self.R_tensor = None

        self.net_name = None
        self.net = None  # neural networks \phi
        self.lstm_net = lstm.net

        self.trainer = None
        self.optimizer_name = None

        self.ae_net = None  # autoencoder networks for pretraining
        self.ae_trainer = None
        self.ae_optimizer_name = None

        self.results = {
            'train_time': None,
            'test_auc': None,
            'test_time': None,
            'test_scores': None,
            'test_score': None,
            'test_fscore': None,
            'test_ftr': None,
            'test_tpr': None,
        }

    def set_network(self, net_name):
        """Builds the neural networks ."""

        self.net_name = net_name
        self.net = build_network(net_name, self.n_code)

    def train(self, dataset: BaseADDataset, optimizer_name: str = 'adam', lr: float = 0.001, n_epochs: int = 50,
              lr_milestones: tuple = (), batch_size: int = 128, weight_decay: float = 1e-6, device: str = 'cuda',
              n_jobs_dataloader: int = 0):
        """Trains the Deep SVDD model on the training data."""

        self.optimizer_name = optimizer_name
        self.trainer = DeepSVDDTrainer(self.objective,
                                       self.R,
                                       self.c,
                                       self.nu,
                                       optimizer_name,
                                       lr=lr,
                                       n_epochs=n_epochs,
                                       lr_milestones=lr_milestones,
                                       batch_size=batch_size,
                                       weight_decay=weight_decay,
                                       device=device,
                                       n_jobs_dataloader=n_jobs_dataloader)
        # Get the model
        self.net = self.trainer.train(dataset, self.net)
        self.R = float(self.trainer.R.cpu().data.numpy())  # get float
        self.c = self.trainer.c.cpu().data.numpy().tolist()  # get list

        self.R_tensor = self.trainer.R
        self.c_tensor = self.trainer.c

        self.results['train_time'] = self.trainer.train_time

    def test(self, dataset: BaseADDataset, device: str = 'cuda', n_jobs_dataloader: int = 0):
        """Tests the Deep SVDD model on the test data."""

        if self.trainer is None:
            self.trainer = DeepSVDDTrainer(objective=self.objective,
                                           R=self.R,
                                           c=self.c,
                                           nu=self.nu,
                                           device=device,
                                           n_jobs_dataloader=n_jobs_dataloader)

        self.trainer.test(dataset, net1=self.lstm_net, net2=self.net)

        # Get results
        self.results['test_auc'] = self.trainer.test_auc
        self.results['test_time'] = self.trainer.test_time
        self.results['test_scores'] = self.trainer.test_scores
        self.results['test_score'] = self.trainer.test_score
        self.results['test_fscore'] = self.trainer.test_fscore
        self.results['test_ftr'] = self.trainer.test_ftr
        self.results['test_tpr'] = self.trainer.test_tpr

    def pretrain(self, dataset: BaseADDataset, optimizer_name: str = 'adam', lr: float = 0.001, n_epochs: int = 100,
                 lr_milestones: tuple = (), batch_size: int = 128, weight_decay: float = 1e-6, device: str = 'cuda',
                 n_jobs_dataloader: int = 0):
        """Pretrains the weights for the Deep SVDD networks via autoencoder."""

        self.ae_net = build_autoencoder(self.net_name)
        self.ae_optimizer_name = optimizer_name
        self.ae_trainer = AETrainer(optimizer_name, lr=lr, n_epochs=n_epochs, lr_milestones=lr_milestones,
                                    batch_size=batch_size, weight_decay=weight_decay, device=device,
                                    n_jobs_dataloader=n_jobs_dataloader)
        self.ae_net = self.ae_trainer.train(dataset, self.ae_net)
        self.ae_trainer.test(dataset, self.ae_net)
        self.init_network_weights_from_pretraining()

    def init_network_weights_from_pretraining(self):
        """Initialize the Deep SVDD networks weights from the encoder weights of the pretraining autoencoder."""
        # Ajoy net_dict应该是DSVDD网络结构（8-16-32-64-32）的参数
        net_dict = self.net.state_dict()
        # Ajoy ae_net_dict应该是DSVDD预训练AE（8-16-32-64-32-64-32-16-（全连接）8）的网络结构参数
        ae_net_dict = self.ae_net.state_dict()

        # Filter out decoder networks keys
        ae_net_dict = {k: v for k, v in ae_net_dict.items() if k in net_dict}
        # Overwrite values in the existing state_dict
        net_dict.update(ae_net_dict)
        # Load the new state_dict
        self.net.load_state_dict(net_dict)

    def save_model(self, export_model, save_ae=True):
        """Save Deep SVDD model to export_model."""
        # Ajoy state_dict()获取网络的所有参数包括计算中的参数，包括bias和weights（但是DSVDD网络中 不存在bias）
        net_dict = self.net.state_dict()
        ae_net_dict = self.ae_net.state_dict() if save_ae else None
        # Ajoy 将所有参数包括半径R、球心C、预训练网络AE参数和DSVDD网络参数
        torch.save({'R': self.R,
                    'c': self.c,
                    'net_dict': net_dict,
                    'ae_net_dict': ae_net_dict}, export_model)

    def load_model(self, model_path, load_ae=False):
        """Load Deep SVDD model from model_path."""

        model_dict = torch.load(model_path)

        self.R = model_dict['R']
        self.c = model_dict['c']
        self.net.load_state_dict(model_dict['net_dict'])
        if load_ae:
            if self.ae_net is None:
                self.ae_net = build_autoencoder(self.net_name)
            self.ae_net.load_state_dict(model_dict['ae_net_dict'])

    def save_results(self, export_json):
        """Save results dict to a JSON-file."""
        with open(export_json, 'w') as fp:
            json.dump(self.results, fp)
