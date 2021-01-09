#include <fstream>
#include <stdlib.h>
#include <stdio.h>
#include <string.h>
#include <sys/time.h>

__global__ void im2colOnDevice(unsigned int n, float *matAc, float *matA, int radiusF, int countF, int L, int M, int K, int C,int H)
{
    for (int idx = blockIdx.x * blockDim.x + threadIdx.x; idx < n; idx += blockDim.x * gridDim.x) {
        int m = (idx / C) / L;
        int l = (idx / C) % L;
        int r = idx % C;
        if (m < M) {
            int w = m + radiusF;
            if (l < L) {
                int h = l + radiusF;
                for (int q = 0, oq = -1 * radiusF; oq <= radiusF; q++, oq++) {
                    for (int p = 0, op = -1 * radiusF; op <= radiusF; p++, op++) {
                        if (r < C) {
                            matAc[(r + C * (p + K * q)) + countF * (l + L * m)] = matA[r + C * ((h + op) + H * (w + oq))]; 
                        }
                    }
                }
            }
        }
    }
}

__global__ void gemm_gpu(double *a,double *b, double *c, int m, int n, int k)
{ 
   int row = blockIdx.y * blockDim.y + threadIdx.y; 
   int col = blockIdx.x * blockDim.x + threadIdx.x;
   double tmp = 0;
   if( col < k && row < m) {
       for(int i = 0; i < n; i++) {
           tmp += a[row * n + i] * b[i * k + col];
       }
       c[row * k + col] = tmp;
   }
}

int main(int argc, char const *argv[])
{
    int W=atoi(argv[1]);
    int H=atoi(argv[2]);
    int C=atoi(argv[3]);
    int K=C;
    int blockSize = 256;
    int gridSize = 0;
    float time_gpu;
    cudaEvent_t start, stop;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);

    int L = H - (K - 1);
    int M = W - (K - 1);
    int KERNELS_NUM = L * M * C;

    int countA = H*W*C;
    const size_t sizeA = countA*sizeof(float);

    int radiusF = (K - 1) / 2;
    int countF = K*K*C;
    int countLR = L * M;
    int countAc = countF * countLR;
    const size_t sizeAc = countAc*sizeof(float);

    float *matA = (float *)malloc(sizeA);
    srand((unsigned)time(0));
    for (int i = 0; i < countA; i++) {
        matA[i] = rand()%10;
    }
    float *devA, *devAc, *retAc;
    cudaMalloc((void**)&devA, sizeA); 
    cudaMalloc((void**)&devAc, sizeAc); 
    retAc = (float *)malloc(sizeAc);
    cudaMemcpy(devA, matA, sizeA, cudaMemcpyHostToDevice); 

    if (gridSize == 0)
        gridSize = (KERNELS_NUM + blockSize - 1) / blockSize;
    
    cudaEventRecord(start, 0);
    im2colOnDevice<<<gridSize, blockSize>>>(KERNELS_NUM, devAc, devA, radiusF, countF, L, M, K, C,H);    
    cudaEventRecord(stop, 0);
    cudaEventSynchronize(stop);
    cudaEventElapsedTime(&time_gpu, start, stop);
    cudaMemcpy(retAc, devAc, sizeAc, cudaMemcpyDeviceToHost);
    printf("共用时间%f ms",time_gpu);

    cudaFree(devA);
    cudaFree(devAc);
    free(matA);
    free(retAc);
    return 0;
}