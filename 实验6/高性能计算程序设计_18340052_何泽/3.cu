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
    // 定义状态变量
    cublasStatus_t status;
    // 在 内存 中为将要计算的矩阵开辟空间
    double *host_A = (double*)malloc (N*M*sizeof(double));
    double *host_B = (double*)malloc (N*M*sizeof(double));
    // 在 内存 中为将要存放运算结果的矩阵开辟空间
    double *host_C = (double*)malloc (M*M*sizeof(double));
    // 为待运算矩阵的元素赋予 0-10 范围内的随机数
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
    // 创建并初始化 CUBLAS 库对象
    cublasHandle_t handle;
    status = cublasCreate(&handle);
    if (status != CUBLAS_STATUS_SUCCESS){
        if (status == CUBLAS_STATUS_NOT_INITIALIZED) {
            cout << "CUBLAS 对象实例化出错" << endl;
        }
        return EXIT_FAILURE;
    }
    double *device_A, *device_B, *device_C;
    // 在 显存 中为将要计算的矩阵开辟空间
    cudaMalloc (
        (void**)&device_A,    // 指向开辟的空间的指针
        N*M * sizeof(double)    //　需要开辟空间的字节数
    );
    cudaMalloc (
        (void**)&device_B,
        N*M * sizeof(double)
    );
    // 在 显存 中为将要存放运算结果的矩阵开辟空间
    cudaMalloc (
        (void**)&device_C,
        M*M * sizeof(double)
    );

    cudaEvent_t start, stop;
    float time;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);
    cudaEventRecord(start, 0);

    // 将矩阵数据传递进 显存 中已经开辟好了的空间
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
    // 同步函数
    cudaThreadSynchronize();
    // 传递进矩阵相乘函数中的参数，具体含义请参考函数手册。
    double a=1; double b=0;
    // 矩阵相乘。该函数必然将数组解析成列优先数组
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
    // 同步函数
    cudaThreadSynchronize();
    // 从 显存 中取出运算结果至 内存中去
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

    // 清理掉使用过的内存
    free (host_A);
    free (host_B);
    free (host_C);
    cudaFree (device_A);
    cudaFree (device_B);
    cudaFree (device_C);
    // 释放 CUBLAS 库对象
    cublasDestroy (handle);
    return 0;
}