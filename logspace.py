
import numpy as np
start = -2
stop = -5
num = 50
x = np.logspace(start, stop, num=num)
print (x)
print (2 ** 4)
import matplotlib.pyplot as plt
plt.plot(x, 'o')
plt.ylim([0.00001, 0.01])
plt.show()