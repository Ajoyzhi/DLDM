import torch
import torch.utils.data.dataset as Dataset
import numpy as np
import torch

np_data = np.zeros((10, 8))
np_label = np.zeros(10)
cc = [[0.5, 1.671291953884821e-06, 0.0019607843137254936, 0.0019607843137254637, 0.0, 1.0, 0.9960629921259841, 0.0]
    , [0.5, 1.671291953884821e-06, 0.0019607843137254936, 0.0019607843137254637, 0.0, 1.0, 0.9960629921259841, 0.0]]
dd = [0, 0, 0, 0, 0, 0, 0, 0, 0, 0]

np_data[0][0] = cc[0][0]
np_data[0][1] = cc[0][1]
np_data[0][2] = cc[0][2]
np_data[0][3] = cc[0][3]
np_data[0][4] = cc[0][4]
np_data[0][5] = cc[0][5]
np_data[0][6] = cc[0][6]
np_data[0][7] = cc[0][7]

np_label[0] = dd[0]
np_label[1] = dd[1]
np_label[2] = dd[2]
np_label[3] = dd[3]
np_label[4] = dd[4]
np_label[5] = dd[5]


class Kdd99(Dataset.Dataset):

    def __init__(self, Data, Label):
        self.Data = torch.Tensor(Data)
        self.Label = torch.Tensor(Label)

    def __getitem__(self, index):
        return self.Data[index], self.Label[index], index

    def __len__(self):
        return len(self.Data)


data = torch.Tensor(np_data)
label = torch.Tensor(np_label)

dataset = Kdd99(data, label)
inputs, labels, _ = dataset.__getitem__(0)

print('data[0]: ', inputs)
print('labels[0]: ', labels)
print(labels.shape)
print(data.shape)
print('len os tensor_dataset: ', len(data))
