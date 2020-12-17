#include <iostream>
#include <cstdio>
#include "cuda_runtime.h"
#include "cublas_v2.h"
using namespace std;
#define isprint 0

int main(int argc, char const *argv[])
{
    int M=atoi(argv[1]);
    int N=atoi(argv[2]);
    cublasStatus_t status;
    double *host_A = (double*)malloc (N*M*sizeof(double));
    double *host_B = (double*)malloc (N*M*sizeof(double));
    double *host_C = (double*)malloc (M*M*sizeof(double));
    srand((unsigned)time(0));
    for (int i=0; i<N*M; i++) {
        host_A[i] = (double)rand() / (double)(RAND_MAX)*1e4;
        host_B[i] = (double)rand() / (double)(RAND_MAX)*1e4;
        // host_A[i] = (int)rand()%10;
        // host_B[i] = (int)rand()%10;
    }
    if(isprint){
        cout << "矩阵 A :" << endl;
        for (int i=0; i<N*M; i++){
            cout << host_A[i] << " ";
            if ((i+1)%N == 0) 
                cout << endl;
        }
        cout << endl;
        cout << "矩阵 B :" << endl;
        for (int i=0; i<N*M; i++){
            cout << host_B[i] << " ";
            if ((i+1)%M == 0)
                cout << endl;
        }
        cout << endl;
    }
    cublasHandle_t handle;
    status = cublasCreate(&handle);
    if (status != CUBLAS_STATUS_SUCCESS){
        if (status == CUBLAS_STATUS_NOT_INITIALIZED) {
            cout << "CUBLAS 对象实例化出错" << endl;
        }
        return EXIT_FAILURE;
    }
    double *device_A, *device_B, *device_C;
    cudaMalloc ((void**)&device_A,N*M * sizeof(double));
    cudaMalloc ((void**)&device_B,N*M * sizeof(double));
    cudaMalloc ((void**)&device_C,M*M * sizeof(double));

    cudaEvent_t start, stop;
    float time;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);
    cudaEventRecord(start, 0);

    cublasSetVector (
        N*M,    // 要存入显存的元素个数
        sizeof(double),    // 每个元素大小
        host_A,    // 主机端起始地址
        1,    // 连续元素之间的存储间隔
        device_A,    // GPU 端起始地址
        1    // 连续元素之间的存储间隔
    );
    cublasSetVector (
        N*M,
        sizeof(double),
        host_B,
        1,
        device_B,
        1
    );
    cudaThreadSynchronize();
    double a=1; double b=0;
    cublasDgemm (
        handle,    // blas 库对象
        CUBLAS_OP_T,    // 矩阵 A 属性参数
        CUBLAS_OP_T,    // 矩阵 B 属性参数
        M,    // A, C 的行数
        M,    // B, C 的列数
        N,    // A 的列数和 B 的行数
        &a,    // 运算式的 α 值
        device_A,    // A 在显存中的地址
        N,    // lda
        device_B,    // B 在显存中的地址
        M,    // ldb
        &b,    // 运算式的 β 值
        device_C,    // C 在显存中的地址(结果矩阵)
        M    // ldc
    );
    cudaThreadSynchronize();
    cublasGetVector (
        M*M,    //  要取出元素的个数
        sizeof(double),    // 每个元素大小
        device_C,    // GPU 端起始地址
        1,    // 连续元素之间的存储间隔
        host_C,    // 主机端起始地址
        1    // 连续元素之间的存储间隔
    );
    cudaEventRecord(stop, 0);
    cudaEventSynchronize(stop);
    cudaEventElapsedTime(&time, start, stop);
    if(isprint){
        cout << "计算结果的转置：" << endl;
        for (int i=0;i<M*M; i++){
            cout << host_C[i] << " ";
            if ((i+1)%M == 0)
                cout << endl;
        }
    }
    printf("\n矩阵A维度%dx%d，矩阵B维度%dx%d，使用CUBLAS运行时间: %f ms.\n", M, N, N, M, time);

    free (host_A);
    free (host_B);
    free (host_C);
    cudaFree (device_A);
    cudaFree (device_B);
    cudaFree (device_C);
    cublasDestroy (handle);
    return 0;
}