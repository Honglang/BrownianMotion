#include <stddef.h>  // NULL, size_t
#include <math.h> // expf
#include <stdio.h> // printf
#include <time.h> // time
#include <sys/time.h> // gettimeofday
#include <assert.h>



#include <curand.h>
#include <cuda.h>
#include <curand_kernel.h>
//#include "cutil.h" // CUDA_SAFE_CALL, CUT_CHECK_ERROR


#include <iostream>
#include <fstream>


using namespace std;




#define NUM_BLOCKS 1000
#define NUM_THREADS 1000  // threads per block
#define N 1000
#define DT 0.001


#define SEED (time(NULL)) // random seed





/***
 * Device functions
 ***/


__global__ void initialize(float *dev_W0){

  int tid = threadIdx.x;
  int bid = blockIdx.x;
  int tt = bid*NUM_BLOCKS+tid;

  dev_W0[tt] = 0.0;
}


__global__ void update_W(float *dev_W0, float *dev_W){

  int tid = threadIdx.x;
  int bid = blockIdx.x;
  int tt = bid*NUM_BLOCKS+tid;

  dev_W0[tt] = dev_W0[tt]+sqrt(DT)*dev_W[tt];

}


/*
main function
*/

int main(void)
{
  
  int i,j;

  curandGenerator_t gen;
  curandCreateGenerator(&gen, CURAND_RNG_PSEUDO_MTGP32);
  curandSetPseudoRandomGeneratorSeed(gen,1234ULL);


  // this is for timing
  clock_t starttime=clock();

  // allocate memory
  float **W;
	W = (float**)malloc(N*sizeof(float*));
  if (W==NULL)
    {
      printf("Can't memalloc W\n");
      return 0;
    }
  
  for (i=0; i<N; i++)
    {
      W[i]=(float*)malloc((NUM_BLOCKS*NUM_THREADS)*sizeof(float));
      if (W[i]==NULL)
        {
          printf("Can't memalloc W[%d]\n",i);
          return 0;
        }
    }



  float *dev_W;
  cudaMalloc( (void**)&dev_W, NUM_BLOCKS*NUM_THREADS*sizeof(float) );
  
  float *dev_W0;
  cudaMalloc( (void**)&dev_W0, NUM_BLOCKS*NUM_THREADS*sizeof(float) );

  initialize<<<NUM_BLOCKS,NUM_THREADS>>>(dev_W0);
  cudaMemcpy(W[0], dev_W0, NUM_BLOCKS*NUM_THREADS*sizeof(float), cudaMemcpyDeviceToHost);
  for(i=1;i<N;i++){
    curandGenerateNormal(gen, dev_W, NUM_BLOCKS*NUM_THREADS, 0.0,1.0);
    update_W<<<NUM_BLOCKS,NUM_THREADS>>>(dev_W0, dev_W);
    cudaMemcpy(W[i], dev_W0, NUM_BLOCKS*NUM_THREADS*sizeof(float), cudaMemcpyDeviceToHost);
  }



  cudaFree(dev_W0);
  cudaFree(dev_W);

  printf("Time elapsed: %f \n",  ((double)clock() - starttime)/CLOCKS_PER_SEC);
    


  printf("Saving a few paths to a file ... \n");
  int myidx[10] = {12, 3234, 534534, 534, 45345, 3434, 999999, 13135, 38, 89343};
  FILE *file1;
  FILE *file2;

  file1 = fopen("path_matrix.txt","w");
  for(j=0;j<N;j++){
    for(i=0;i<10;i++){
        fprintf(file1,"%.6f ", W[j][ myidx[i] ]);
    }
    fprintf(file1, "\n");
  }
  fclose(file1);


  // check to see if W(0.5) and W(1) have the correct distributions
  file2 = fopen("for_hist.txt","w");
  for(j=0;j<1000000;j++){
    fprintf(file2, "%.6f, ", W[499][j]);
    fprintf(file2, "%.6f\n", W[999][j]);
  }
  fclose(file2);


  free(W);

  return 1;
}