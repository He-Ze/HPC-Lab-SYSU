#include <stdio.h>
#include <stdlib.h>
#define isprint 0

__global__ void Conv2DKernel(float *output, float *input, float *kernel, int inputSize, int kernelSize)
{
    int col = threadIdx.x + blockDim.x * blockIdx.x;
    int row = threadIdx.y + blockDim.y * blockIdx.y;

    const int limit = inputSize - kernelSize + 1;
    if (col >= limit || row >= limit)
        return;
    int curCol = 0;
    int curRow = 0;
    float sum = 0.0f;
    for (int i = 0; i < kernelSize; ++i){
        for (int j = 0; j < kernelSize; ++j){
            curCol = col + j;
            curRow = row + i;
            sum += (kernel[i * kernelSize + j] * input[curRow * inputSize + curCol]);
        }
    }
    output[row * limit + col] = sum;
}

void display(float *arr, int w, int h)
{
    for (int i = 0; i < w; ++i){
        for (int j = 0; j < h; ++j){
            printf("%d\t", int(arr[i * w + j]));
        }
        printf("\n");
    }
    printf("\n");
}

int rand_num(int start, int end)
{
    return rand() % (end + 1 - start) + start;
}

int main(int argc, char const *argv[])
{
    int W=atoi(argv[1]);
    int H=atoi(argv[2]);
    int KERNEL_SIZE=atoi(argv[3]);
    int TB=KERNEL_SIZE;

    int imgSize = W * H;
    int convOutW = W - KERNEL_SIZE + 1;
    int convOutSize = convOutW * convOutW;
    int mSize = imgSize * sizeof(float);

    float time_gpu;
    cudaEvent_t start, stop;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);

    float *h_A = (float *)malloc(mSize);
    float *h_Kernel = (float *)malloc(KERNEL_SIZE * KERNEL_SIZE * sizeof(float));
    float *h_C = (float *)malloc(convOutSize * sizeof(float));

    srand((unsigned)time(0));
    for (int i = 0; i < imgSize; ++i){
        h_A[i] = rand_num(0, 100);
    }
    for (int j = 0; j < KERNEL_SIZE * KERNEL_SIZE; ++j){
        h_Kernel[j] = rand_num(0, 5);
    }
    if(isprint){
        printf("矩阵A （%d行%d列）：\n",W,H);
        display(h_A, W, H);
        printf("Kernel （%d行%d列）：\n",KERNEL_SIZE,KERNEL_SIZE);
        display(h_Kernel, KERNEL_SIZE, KERNEL_SIZE);
    }

    float *d_A = NULL;
    float *d_Kernel = NULL;
    cudaMalloc(&d_A, mSize);
    cudaMalloc(&d_Kernel, KERNEL_SIZE * KERNEL_SIZE * sizeof(float));

    cudaMemcpy(d_A, h_A, mSize, cudaMemcpyHostToDevice);
    cudaMemcpy(d_Kernel, h_Kernel, KERNEL_SIZE * KERNEL_SIZE * sizeof(float), cudaMemcpyHostToDevice);

    float *d_C = NULL;
    cudaMalloc(&d_C, convOutSize * sizeof(float));

    dim3 dimBlock(TB, TB);
    int tb = (W + TB - 1) / TB;
    dim3 dimGrid(tb, tb);

    cudaEventRecord(start, 0);
    Conv2DKernel<<<dimGrid, dimBlock>>>(d_C, d_A, d_Kernel, W, KERNEL_SIZE);
    cudaEventRecord(stop, 0);
    cudaEventSynchronize(stop);
    cudaEventElapsedTime(&time_gpu, start, stop);

    cudaMemcpy(h_C, d_C, convOutSize * sizeof(float), cudaMemcpyDeviceToHost);
    if(isprint){
        printf("结果 （%d行%d列）：\n",convOutW,convOutW);
        display(h_C, convOutW, convOutW);
    }

    printf("共用时间%f ms",time_gpu*10);
    
    cudaFree(d_A);
    cudaFree(d_Kernel);
    cudaFree(d_C);
    free(h_A);
    free(h_Kernel);
    free(h_C);
    return 0;
}