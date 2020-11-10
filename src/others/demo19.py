import numpy as np

x = np.array([6, 17, 17, 17, 6, 17, 6])
x = (x - np.min(x)) / (np.max(x) - np.min(x))

print(x)

