import numpy as np
import pandas as pd
import csv

csv_reader = [
    [0, 0, 47, 5, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 146, 16, 1.00, 1.00, 0.00, 0.00, 0.11, 0.05,
     0.00, 255, 16, 0.06, 0.06, 0.00, 0.00, 1.00, 1.00, 0.00, 0.00, 4],
    [0, 0, 21, 9, 181, 5450, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 8, 8, 0.00, 0.00, 0.00, 0.00, 1.00, 0.00,
     0.00, 9, 9, 1.00, 0.00, 0.11, 0.00, 0.00, 0.00, 0.00, 0.00, 0],
    [0, 0, 21, 9, 285, 765, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 36, 36, 0.00, 0.00, 0.00, 0.00, 1.00, 0.00,
     0.00, 255, 255, 1.00, 0.00, 0.00, 0.00, 0.00, 0.00, 0.00, 0.00, 0]
]

count = 0  # 行数

data_file = open("C:/Users/Administrator/Desktop/aa.txt", 'w')
csv_writer = csv.writer(data_file)
for row in csv_reader:  # 循环读取文件数据
    temp_line = np.array(row)
    temp_line[1] = 1
    temp_line[2] = 2
    temp_line[3] = 4
    if temp_line[41] != 0:  # 不写
        print("不等于0的数据不用写")
        continue
    count += 1
    print(count, temp_line)
    csv_writer.writerow(temp_line)

print(0 != 0)