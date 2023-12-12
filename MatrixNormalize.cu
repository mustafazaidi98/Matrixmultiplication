#include <stdio.h>
#include <time.h>
#include <sys/time.h>
#include <cuda_runtime.h>
#include <helper_cuda.h>
__global__ void matrixNormalize(const float *A, float *C,float *R,int numElements) {
  int col = blockDim.x * blockIdx.x + threadIdx.x;
  float avg = 0;
  float std = 0;
  float prev =  A[0+col];
  if (col < numElements) {
        C[col] = 0;
        for(int i=0; i<numElements*numElements-1;i+=numElements)
        {     
		// if(prev!=A[i+col])
                //        printf("Distrupcy, thread:%d, value:%f\n",col,A[i+col]);
                C[col]+=A[i+col];
                prev = A[i+col];
        }
   avg = C[col]/numElements;
 // printf("thread %d,block: %d avg =  %f, sum = %f\n",col,blockIdx.x,avg,C[col]);
        for(int i=0; i<numElements*numElements;i+=numElements){
                std += powf(A[i+col] - avg,2.0);
        }
  std /= (float)numElements;
  std = sqrt(std);
  //printf("thread %d, calculated std as  %f\n",col,std);
  for(int i=0;i<numElements*numElements-1;i+=numElements){
                if(std != 0.0)
                        R[i+col] = (A[i+col]-avg)/std;
                else
                        R[i+col] = 0.0;
        }
  }
else{
 // printf("I'm thread %d, from block %float *h_C = (float *)malloc(numElements* sizeof(float));d didnt>  }

}
}
int numElements;
int main(int argc, char **argv) {
 struct timeval start,stop;
 struct timezone tzdummy;
 unsigned long long runtime;
int printMt =  0;
  // Error code to check return values for CUDA calls
  cudaError_t err = cudaSuccess;
  if(argc > 1)
{
        numElements = atoi(argv[1]);
}
else
  numElements =  10000;
if(argc >2)
  printMt =  atoi(argv[2]);
  size_t size = numElements * numElements * sizeof(float);
  printf("[Matrix Normalization of %d rows and columns]\n", numElements);

  // Allocate the host input vector A
  float *h_A = (float *)malloc(size);

  // Allocate the host output vector C
  float *h_C = (float *)malloc(numElements* sizeof(float));

  // Allocate the host for result vector R
  float *h_R = (float *) malloc(size);
  // Verify that allocations succeeded
  if (h_A == NULL || h_C == NULL| h_R==NULL) {
    fprintf(stderr, "Failed to allocate host vectors!\n");
    exit(EXIT_FAILURE);
  }
   int val = 0;
   srand((unsigned)time(NULL));
  // Initialize the host input vectors
  for (int i = 0; i < numElements*numElements; i++) {
    h_A [i] = (float)rand()/32768.0;
   }
  // Allocate the device input vector A
  float *d_A = NULL;
  err = cudaMalloc((void **)&d_A, size);

  if (err != cudaSuccess) {
    fprintf(stderr, "Failed to allocate device vector A (error code %s)!\n",
            cudaGetErrorString(err));
    exit(EXIT_FAILURE);
  }

  // Allocate the device output vector C
  float *d_C = NULL;
  err = cudaMalloc((void **)&d_C, numElements*sizeof(float));

  if (err != cudaSuccess) {
    fprintf(stderr, "Failed to allocate device vector C (error code %s)!\n",
            cudaGetErrorString(err));
    exit(EXIT_FAILURE);
 }
  // Allocate the device for result vector R
  float *d_R = NULL;
  err = cudaMalloc((void **)&d_R, size);
  if (err != cudaSuccess){
  fprintf(stderr,"Failed to allocate device for result vector R\n");
  exit(EXIT_FAILURE);
        }
  // Copy the host input vectors A and B in host memory to the device input
  // vectors in
  // device memory
 printf("\n----------------\n");
 printf("Matrix size N = %d",numElements);
 printf("\nStarting Clock\n\n");
// gettimeofday(&start,&tzdummy);
  printf("Copy input data from the host memory to the CUDA device\n");
  err = cudaMemcpy(d_A, h_A, size, cudaMemcpyHostToDevice);

  if (err != cudaSuccess) {
    fprintf(stderr,
            "Failed to copy vector A from host to device (error code %s)!\n",
            cudaGetErrorString(err));
    exit(EXIT_FAILURE);
  }


  // Launch the Vector Add CUDA Kernel
  int threadsPerBlock = 256;
  int blocksPerGrid = (numElements + threadsPerBlock -1 )/ threadsPerBlock;
  printf("CUDA kernel launch with %d blocks of %d threads\n", blocksPerGrid,
         threadsPerBlock);
 printf("Total columns: %d, total threads: %d\n",numElements,threadsPerBlock*blocksPerGrid);
  gettimeofday(&start,&tzdummy);
  matrixNormalize<<<blocksPerGrid, threadsPerBlock>>>(d_A, d_C,d_R, numElements);
  gettimeofday(&stop,&tzdummy);
  err = cudaGetLastError();

  if (err != cudaSuccess) {
    fprintf(stderr, "Failed to launch vectorAdd kernel (error code %s)!\n",
            cudaGetErrorString(err));
    exit(EXIT_FAILURE);
  }

  // Copy the device result vector in device memory to the host result vector
  // in host memory.
  printf("Copy output data from the CUDA device to the host memory\n");
  err = cudaMemcpy(h_R, d_R, size, cudaMemcpyDeviceToHost);

  if (err != cudaSuccess) {
    fprintf(stderr,
            "Failed to copy vector C from device to host (error code %s)!\n",
            cudaGetErrorString(err));
    exit(EXIT_FAILURE);
  }
  //gettimeofday(&stop,&tzdummy);
 runtime = (unsigned long long)(stop.tv_sec - start.tv_sec) * 1000000 + (stop.tv_usec - start.tv_usec);
 printf("Runtime: %g ms.\n",(float)runtime/(float)1000);
  printf("Test PASSED\n");

  // Free device global memory
  err = cudaFree(d_A);

  if (err != cudaSuccess) {
    fprintf(stderr, "Failed to free device matrix A (error code %s)!\n",
            cudaGetErrorString(err));
    exit(EXIT_FAILURE);
  }

  err = cudaFree(d_C);

  if (err != cudaSuccess) {
    fprintf(stderr, "Failed to free device vector C (error code %s)!\n",
            cudaGetErrorString(err));
    exit(EXIT_FAILURE);
  }

  if(err != cudaSuccess){
   fprintf(stderr,"Failed to free device of matrix");
  exit(EXIT_FAILURE);
}
if(printMt==1)
{
printf("\n----Input Vector----\n");
 for(int i=0; i< numElements*numElements ; i+=1)
{
  if(i%numElements==0 && i!=0)
        printf("\n");
   printf("%f      ",h_A[i]);
}
printf("\n----Output Vector----\n");
for(int i=0;i<numElements*numElements; i+=1){
 if(i%numElements ==  0 )
        printf("\n");
 printf("%f     ",h_R[i]);
}
}
  // Free host memory
  free(h_A);
  free(h_C);
  free(h_R);
  printf("Done\n");
  return 0;
}


