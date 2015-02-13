import numpy as np
import math
from timeit import default_timer as timer  # for timing

WW=np.zeros([1000000,1000])

T=1
N=1000
tt=np.linspace(0,T,num=N)
dt = T / float(N)
W=np.zeros(N)

ts = timer()	
for k in range(1000000):
	W[1:N]=np.sqrt(dt)*np.random.normal(0,1,N-1)
	WW[k,:]=np.cumsum(W)

te=timer()
fmt = '%20s: %s'
print fmt % ('time elapsed', '%.5fs' % (te - ts))
