#include <stdio.h>
#include <stdlib.h>
#include <omp.h>
#define RUN_CPU 0

int block_size=8,threads,m,n,k;
double *host_a, *host_b, *host_c_gpu, *host_c_cpu;
double *device_a, *device_b, *device_c;
float time_gpu, time_cpu;

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

void gemm_cpu(double *host_a, double *host_b, double *host_result, int m, int n, int k) 
{
    for (int i = 0; i < m; i++) {
        for (int j = 0; j < k; j++) {
            double tmp = 0.0;
            for (int h = 0; h < n; h++) {
                tmp += host_a[i * n + h] * host_b[h * k + j];
            }
            host_result[i * k + j] = tmp;
        }
    }
}

void mul_one_block (int upperOfRow,int bottomOfRow,int leftOfCol,int rightOfCol,int transLeft,int transRight)
{
    int mm=bottomOfRow-upperOfRow;
    int nn=rightOfCol-leftOfCol;
    int kk=transRight-transLeft;
    cudaMalloc((void **) &device_a, sizeof(double)*mm*nn);
    cudaMalloc((void **) &device_b, sizeof(double)*nn*kk);
    cudaMalloc((void **) &device_c, sizeof(double)*mm*kk);
    cudaMemcpy(device_a, host_a, sizeof(double)*mm*nn, cudaMemcpyHostToDevice);
    cudaMemcpy(device_b, host_b, sizeof(double)*nn*kk, cudaMemcpyHostToDevice);

    unsigned int grid_rows = (mm + block_size - 1) / block_size/threads;
    unsigned int grid_cols = (kk + block_size - 1) / block_size/threads;
    dim3 dimGrid(grid_cols, grid_rows);
    dim3 dimBlock(block_size, block_size);
    gemm_gpu<<<dimGrid, dimBlock>>>(device_a, device_b, device_c, mm, nn, kk);

    cudaMemcpy(host_c_gpu, device_c, sizeof(double)*mm*kk, cudaMemcpyDeviceToHost);
    cudaThreadSynchronize();
}

void omp_plus_cuda(int upperOfRow,int bottomOfRow,int leftOfCol,int rightOfCol,int transLeft,int transRight)
{
    if ( ( bottomOfRow - upperOfRow ) < m/2 ) 
        mul_one_block (int upperOfRow,int bottomOfRow,int leftOfCol,int rightOfCol,int transLeft,int transRight);
    else{
        #pragma omp task{
        matrixMulti( upperOfRow , ( upperOfRow + bottomOfRow ) / 2 ,leftOfCol , ( leftOfCol + rightOfCol ) / 2 ,transLeft , ( transLeft + transRight ) / 2 );
        matrixMulti( upperOfRow , ( upperOfRow + bottomOfRow ) / 2 ,leftOfCol , ( leftOfCol + rightOfCol ) / 2 ,( transLeft + transRight ) / 2 + 1 , transRight );
        }
        #pragma omp task{
        matrixMulti( upperOfRow , ( upperOfRow + bottomOfRow ) / 2 ,( leftOfCol + rightOfCol ) / 2 + 1 , rightOfCol ,transLeft , ( transLeft + transRight ) / 2 );
        matrixMulti( upperOfRow , ( upperOfRow + bottomOfRow ) / 2 ,( leftOfCol + rightOfCol ) / 2 + 1 , rightOfCol ,( transLeft + transRight ) / 2 + 1 , transRight );
        }
        #pragma omp task{
        matrixMulti( ( upperOfRow + bottomOfRow ) / 2 + 1 , bottomOfRow ,leftOfCol , ( leftOfCol + rightOfCol ) / 2 ,transLeft , ( transLeft + transRight ) / 2 );
        matrixMulti( ( upperOfRow + bottomOfRow ) / 2 + 1 , bottomOfRow ,leftOfCol , ( leftOfCol + rightOfCol ) / 2 ,( transLeft + transRight ) / 2 + 1 , transRight );
        }
        #pragma omp task{
        matrixMulti( ( upperOfRow + bottomOfRow ) / 2 + 1 , bottomOfRow ,( leftOfCol + rightOfCol ) / 2 + 1 , rightOfCol ,transLeft , ( transLeft + transRight ) / 2 );
        matrixMulti( ( upperOfRow + bottomOfRow ) / 2 + 1 , bottomOfRow ,( leftOfCol + rightOfCol ) / 2 + 1 , rightOfCol ,( transLeft + transRight ) / 2 + 1 , transRight );
        }
        #pragma omp taskwait
    }
}
 
int main(int argc, char const *argv[])
{
    threads=atoi(argv[1]);
    m=atoi(argv[2]);
    n=atoi(argv[3]);
    k=atoi(argv[4]);
    cudaMallocHost((void **) &host_a, sizeof(double)*m*n);
    cudaMallocHost((void **) &host_b, sizeof(double)*n*k);
    cudaMallocHost((void **) &host_c_gpu, sizeof(double)*m*k);
    cudaMallocHost((void **) &host_c_cpu, sizeof(double)*m*k);

    srand((unsigned)time(0));
    for (int i = 0; i < m; ++i) {
        for (int j = 0; j < n; ++j) {
            host_a[i * n + j] = (double)rand() / (double)(RAND_MAX)*1e4;
        }
    }
    for (int i = 0; i < n; ++i) {
        for (int j = 0; j < k; ++j) {
            host_b[i * k + j] = (double)rand() / (double)(RAND_MAX)*1e4;
        }
    }

    cudaEvent_t start, stop;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);
    cudaEventRecord(start, 0);
    omp_plus_cuda(0,m-1,0,n-1,0,k-1);
    cudaEventRecord(stop, 0);
    cudaEventSynchronize(stop);
    cudaEventElapsedTime(&time_gpu, start, stop);
    printf("矩阵A维度%dx%d，矩阵B维度%dx%d，Block_size为%d，在GPU上运行时间: %f ms.\n", m, n, n, k, block_size, time_gpu);

    if(RUN_CPU){
        cudaEventRecord(start, 0);
        gemm_cpu(host_a, host_b, host_c_cpu, m, n, k);
        cudaEventRecord(stop, 0);
        cudaEventSynchronize(stop);
        cudaEventElapsedTime(&time_cpu, start, stop);
        printf("矩阵A维度%dx%d，矩阵B维度%dx%d，Block_size为%d，在CPU上运行时间: %f ms.\n\n", m, n, n, k, block_size, time_cpu);
        int all_ok = 1;
        for (int i = 0; i < m; ++i){
            for (int j = 0; j < k; ++j){
                if(host_c_cpu[i*k + j] != host_c_gpu[i*k + j]){
                    all_ok = 0;
                }
            }
        }
        if(all_ok){
            printf("结果正确，加速比为%f\n", time_cpu / time_gpu);
        }
        else{
            printf("结果错误\n");
        }
    }
    cudaFree(device_a);
    cudaFree(device_b);
    cudaFree(device_c);
    cudaFreeHost(host_a);
    cudaFreeHost(host_b);
    cudaFreeHost(host_c_gpu);
    cudaFreeHost(host_c_cpu);
    return 0;
}