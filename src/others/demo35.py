
import numpy as np
cols = 4
rows = 4

list = [[0 for col in range(cols)] for row in range(rows)]
a = np.array([123123,12312,31,23,12,3,12,3,123])

print(list)
list[0].append('aa')
list[0].append(a)
print(list[0][5])


list = []
list.append(0.0)
list.append(0.0)
list.append(0.0)
list.append(0.0)
print(list)

def ll(list):
    for i in range(4):
        list.append(0.0)
    return list

list[0] += 9.0
list[0] += 1.0
list[0] += 2.0
list[0] += 3.0
print(np.array(list)/4.0)


def cc():
    list = []
    for i in range(4):
        list.append([])
    return list

list = cc()
print(list)
print(list[3])

A = [[] for i in range(4)]
print(A)
A = [0.0 for i in range(4)]
print(A)
A = [None for i in range(4)]
print(A)
## scores, aucs
n = 4
times = [5.0 for i in range(4)]
times = [times[i] / n for i in range(n)]
print(times)



