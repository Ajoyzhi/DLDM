import torch.utils.data.dataset as Dataset
import torch
from datasets.pre_file import pre_file
from datasets.pre_file import get_anomaly_from_train
from datasets.pre_data import load_data_kdd99
from filePaths import src_train
from filePaths import src_test

from filePaths import handle_train
from filePaths import handle_test

from filePaths import final_train
from filePaths import final_test

from filePaths import handle_train_anomaly
from filePaths import final_train_anomaly

from base.torchvision_dataset import TorchvisionDataset

"""
Ajoy
    训练集:  原始 ./dataset/kddcup_10_percent_corrected
            数值化（所有正常数据） ./dataset/kddcup_10_percent_corrected.csv
            选择对应特征（所有正常数据） ./dataset/kddcup.data_10_percent_final.cvs
    测试集： 原始 ./dataset/corrected
            数值化 ./dataset/corrected.cvs
            选择对应特征 ./dataset/corrected_final.cvs
"""
class Kdd99_Dataset(TorchvisionDataset):
    """
        数据集：
            来自于 Kdd99 数据集的九个特征
            该数据集作为 lstm-autoencoder 的输入

        属性：
            train：训练集的数据,       shape = (xxxxx, 9)
            train_label：训练集的标签  shape = (xxxxx,)
            test：测试集的数据         shape = (yyyyy, 9)
            test_label：测试集的标签   shape = (yyyyy,)

            test：测试集的类型，kdd99测试集(0), sdn测试集(1)  ---  已废弃
            n_features：特征数目
            dos_type：dos 攻击种类数

            exper_type：代表实验类型
                0：基础实验（join，ae_kmeans），训练集获取正常数据，测试集获取所有数据
                1：对比实验（rbm）：训练集获取所有数据，测试集获取所有数据
                2：对比实验（join，ae_kmeans，dos_types）：训练集获取正常数据，测试集获取正常数据 + 指定攻击
                3：对比实验（rbm，dos_types）：训练集获取所有数据，测试集获取正常数据 + 指定攻击

    """

    def __init__(self, n_features=8, exper_type=0, dos_types=0):
        """  """
        super().__init__()
        self.n_features = n_features
        self.exper_type = exper_type
        self.dos_types = dos_types
        self.train = None
        self.test = None
        self.train_labels = None
        self.test_labels = None

        #Ajoy 增加获取训练集中异常数据的属性
        self.anomaly_train = None
        self.anomaly_train_label = None

        # Ajoy 调用过程，src_train\handle_train均在filepaths.py中
        pre_file(src_train, handle_train, train=1, exper_type=self.exper_type, dos_types=self.dos_types)
        pre_file(src_test, handle_test, train=0, exper_type=self.exper_type, dos_types=self.dos_types)

        # Ajoy 筛选训练集所有的异常数据
        get_anomaly_from_train(src_train, handle_train_anomaly)

        #AJOY 加载了KDD99中固定的9个特征（同时还将处理后的数据进行了保存） pre_data
        train, train_label = load_data_kdd99(handle_train, final_train, self.n_features)
        test, test_label = load_data_kdd99(handle_test, final_test, self.n_features)  # kdd99 测试集

        # Ajoy 选择训练集异常数据的指定属性
        anomaly, anomaly_label = load_data_kdd99(handle_train_anomaly, final_train_anomaly, self.n_features)

        self.train = train
        self.test = test
        self.train_labels = train_label
        self.test_labels = test_label

        self.anomaly_train = anomaly
        self.anomaly_train_label = anomaly_label

        # AJoy 获取了训练接和测试集
        # Ajoy 通过父类TorchvisionDataset中得到loader方法，loader方法返回训练集和测试集
        self.train_set = Kdd99(train, train_label)
        self.test_set = Kdd99(test, test_label)

        self.anomoly_set = Kdd99(anomaly, anomaly_label)

        print("train", train.shape)
        print("train_label", train_label.shape)
        print("test", test.shape)
        print("test_label", test_label.shape)

        print("anomaly_train", anomaly.shape)
        print("anomaly_train_label", anomaly_label)

        print(self.train_set.__getitem__(0))
        print(self.test_set.__getitem__(0))

        print(self.anomaly_train_label.__getitem__(0))
    # AJoy 为啥更新？
    # Ajoy 是不是对多个数据进行存储？
    def update_test(self, exper_type=0, dos_types=0):
        """ 多次 dos 攻击，更新测试集 """
        pre_file(src_test, handle_test, 0, exper_type=exper_type, dos_types=dos_types)
        test, test_label = load_data_kdd99(handle_test, final_test, self.n_features)  # kdd99 测试集
        self.test_set = Kdd99(test, test_label)
        self.test = test
        self.test_labels = test_label

        print("test", test.shape)
        print("test_label", test_label.shape)


class Kdd99(Dataset.Dataset):

    def __init__(self, Data, Label):
        self.Data = torch.Tensor(Data)
        self.Label = torch.Tensor(Label)

    def __getitem__(self, index):
        return self.Data[index], self.Label[index], index

    def __len__(self):
        return len(self.Data)

