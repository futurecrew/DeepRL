import numpy as np

a = np.zeros((84, 84))
b = np.ones((84, 84))

list = []
list.append(a)
list.append(b)

c = np.reshape(list, (2, 84, 84))

print c.shape
