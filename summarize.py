import numpy as np
import math
import matplotlib.pyplot as plt


BM = np.genfromtxt('path_matrix.txt', delimiter=' ')
HH = np.genfromtxt('for_hist.txt', delimiter=',')

tt=np.linspace(0,1,1000)

print(np.sqrt(0.5))
print(np.std(HH[:,0]))
print(np.std(HH[:,1]))

plt.figure(1)
plt.subplot(121)
plt.hist(HH[:,0], bins=30)
plt.subplot(122)
plt.hist(HH[:,1], bins=30)


plt.figure(2)
for k in range(10):
	plt.plot(tt,BM[:,k])

plt.show()

