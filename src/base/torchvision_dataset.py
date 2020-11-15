from .base_dataset import BaseADDataset
from torch.utils.data import DataLoader

"""
Ajoy
    增加获取训练集异常数据的loader
"""
class TorchvisionDataset(BaseADDataset):
    """TorchvisionDataset class for datasets already implemented in torchvision.datasets."""

    def __init__(self):
        super().__init__()

    def loaders(self, batch_size: int, shuffle_train=True, shuffle_test=False, num_workers: int = 0) -> (
            DataLoader, DataLoader):
        train_loader = DataLoader(dataset=self.train_set, batch_size=batch_size, shuffle=shuffle_train,
                                  num_workers=num_workers)
        test_loader = DataLoader(dataset=self.test_set, batch_size=batch_size, shuffle=shuffle_test,
                                 num_workers=num_workers)
        return train_loader, test_loader

    def anomaly_loaders(self,  batch_size: int, shuffle_train_anomaly=True, num_workers: int = 0) -> (
            DataLoader):

        anomaly_loader = DataLoader(dataset=self.anomaly_set, batch_size=batch_size, shuffle=shuffle_train_anomaly,
                                  num_workers=num_workers)
        return anomaly_loader
