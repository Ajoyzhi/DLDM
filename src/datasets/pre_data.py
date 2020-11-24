import random

import torch
import numpy as np
import csv
import pandas


"""
Ajoy：load_data_kdd99
      输入：handled_file（读取数据的文件）；final_file（输出的文件）；features（抽取的特征数量）
      可以复用
"""
def z_score_normalization(x):
    x = (x - np.mean(x)) / np.std(x)
    return x


def z_score_normalizations(x_mat):
    """ 标准化数据 -- 只考虑连续型数据 -- 第一个proto为离散型 """
    for i in range(1, x_mat.shape[1]):
        x_mat[:, i] = z_score_normalization(x_mat[:, i])

    return x_mat


def min_max_normalization(x):
    x = (x - np.min(x)) / (np.max(x) - np.min(x))
    return x


def min_max_normalizations(x_mat):
    """ 归一化数据 """
    for i in range(x_mat.shape[1]):
        x_mat[:, i] = min_max_normalization(x_mat[:, i])

    return x_mat

"""
Ajoy 
    加入加载的数据比例
    ratio:float类型 [0,1]
"""
def load_data_kdd99(handled_file, final_file, features, ratio):
    """
        数据预处理：
            输入文件名 xx.csv
            输出文件名 xx_final.csv

        选取实验需要的特征，就是下面八个特征。
            1  protocol_type 离散型
            3  src_bytes
            22 count
            23 srv_count
            30 srv_diff_host_rate
            31 dst_host_count
            32 dst_host_srv_count
            37 dst_host_src_diff_host_rate
            xx duration

        标准化处理
            连续型：l2 norm
            离散型：不作处理

        归一化处理
            离散型：最大值最小值归一化
            连续型：最大值最小值归一化

    """
    fr = open(handled_file)
    lines = fr.readlines()
    rows = len(lines)

    x_mat = np.zeros((rows, features))
    y_label = np.zeros(rows)
    # Ajoy 从输入文件中选择对应的特征
    for i in range(rows):
        line = lines[i].strip()
        item_mat = line.split(',')

        # 考虑src_bytes
        if features == 9:
            x_mat[i][0] = item_mat[1]   # protocol_type -- 离散型
            x_mat[i][1] = item_mat[4]   # src_bytes
            x_mat[i][2] = item_mat[22]  # count
            x_mat[i][3] = item_mat[23]  # srv_count
            x_mat[i][4] = item_mat[30]  # srv_diff_host_rate
            x_mat[i][5] = item_mat[31]  # dst_host_count
            x_mat[i][6] = item_mat[32]  # dst_host_srv_count
            x_mat[i][7] = item_mat[36]  # dst_host_src_diff_host_rate
            x_mat[i][8] = item_mat[0]   # duration

        # 不考虑src_bytes
        elif features == 8:
            x_mat[i][0] = item_mat[1]   # protocol_type -- 离散型
            x_mat[i][1] = item_mat[22]  # count
            x_mat[i][2] = item_mat[23]  # srv_count
            x_mat[i][3] = item_mat[30]  # srv_diff_host_rate
            x_mat[i][4] = item_mat[31]  # dst_host_count
            x_mat[i][5] = item_mat[32]  # dst_host_srv_count
            x_mat[i][6] = item_mat[36]  # dst_host_src_diff_host_rate
            x_mat[i][7] = item_mat[0]   # duration

        y_label[i] = item_mat[41]   # label
        # print(i, "-", x_mat[i][0], x_mat[i][1], x_mat[i][2], x_mat[i][3], x_mat[i][4], x_mat[i][5], x_mat[i][6], x_mat[i][7], x_mat[i][8])

    fr.close()
    # Ajoy 对特征进行统一处理（标准化、归一化）
    x_mat = z_score_normalizations(x_mat)
    x_mat = min_max_normalizations(x_mat)

    # print("x_mat:", x_mat.shape)
    # print("y:", y_label.shape)

    # AJOY：从x_mat和y_label中随机选择对应的数量的样本
    # Ajoy 确定选择的样本数量，转换为整数
    number = int(rows * ratio)
    # Ajoy 将数据和label合并为一个数组，输入到文件中
    data_label = np.c_[x_mat,y_label]
    train_sample = sample(data_label, number)

    print("train_simple:", train_sample.shape)

    # AJoy 最终返回数据和label
    x_data = [row[:-1] for row in train_sample]
    y = [row[-1] for row in train_sample]


    #Ajoy 将处理好的数据输入到final文件中统一保存
    write_file(data_label, final_file)
    # Ajoy 返回抽取的特征和label
    return x_data, y

def write_file(data, data_file):
    data_file = open(data_file, 'w', newline='')
    csv_writer = csv.writer(data_file)
    count = 0  # 行数
    for row in data:  # 循环读取文件数据
        temp_line = np.array(row)
        csv_writer.writerow(temp_line)
        count += 1
        # print(count, 'final:', temp_line[0], temp_line[1], temp_line[2], temp_line[3])
    data_file.close()


#功能：随机选择array中固定数量的行
#输入：array:原矩阵；number：选择的数量
def sample(array, number):
    rand_arr = np.arange(array.shape[0])

    np.random.shuffle(rand_arr)
    sample= array[rand_arr[0:number]]

    return sample






