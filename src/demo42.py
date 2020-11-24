"""
Ajoy: Test the results of getting anomaly data from train set
      get the special features from the anomaly data, True
      注意：测试时需要将该文件放于src文件夹下，因为读/写的文件路径皆以src为根目录
"""

# Ajoy 筛选训练集所有的异常数据
from filePaths import handle_train_anomaly, final_train_anomaly, handle_train, final_train, handle_test, \
    final_test
from pre_data import load_data_kdd99

import time
"""
start_time = time.time()
# 读数据集文件，将所有异常数据数值化之后写入目标文件（corrected_anomaly.csv），其中包含标签
get_anomaly_from_train(src_train, handle_train_anomaly)

# 读异常数据的数值化文件，选择固定的9个特征，标准化和归一化，写入到目标文件（anomaly_final.csv），未带标签
load_data_kdd99(handle_train_anomaly, final_train_anomaly, 9)

end_time = time.time()
print("time:", end_time - start_time)"""




start_time = time.time()
# Ajoy 训练集中的正常数据9万多条，只加载10%，大概9千多条
train, train_label = load_data_kdd99(handle_train, final_train, 9, 0.1)
# AJoy 加载全部的测试数据
test, test_label = load_data_kdd99(handle_test, final_test, 9, 1)  # kdd99 测试集

# Ajoy 选择训练集异常数据的指定属性，并30万条数据，加载1%，大概3000条数据
anomaly, anomaly_label = load_data_kdd99(handle_train_anomaly, final_train_anomaly, 9, 0.01)
end_time  = time.time()

print("time:", end_time-start_time)
