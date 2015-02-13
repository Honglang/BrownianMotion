# BrownianMotion

Python and CUDA code for simulating 1000000 BM[0,1] paths with dt=0.001

[Python](bm.py)

[CUDA](bm.cu)

Compile the .cu file with 

```
nvcc -arch sm_13 -L /usr/local/cuda/lib64 -lcurand -o brmotion bm.cu
```

To simulate, type
```
./brmotion
```
