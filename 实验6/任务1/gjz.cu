# include <stdio.h>
# include <stdlib.h>
# include <cuda_runtime.h>

# define TIME_INFO
//# define OUTPUT_RESULT

# ifdef TIME_INFO
# include <sys/time.h>
double wtime() {
    timeval t;
    gettimeofday(&t, NULL);
    return t.tv_sec + t.tv_usec / 1e6;
}
# endif // TIME_INFO

# ifdef OUTPUT_RESULT
void matrixPrint(const double* nums, int m, int n) {
    putchar('[');
    for (int i = 0; i < m; ++i) {
        putchar('[');
        for (int j = 0; j < n; ++j) {
            int index = i * n + j;
            printf("%g", nums[index]);
            printf(j != n - 1? ", ": (i != m - 1? "],\n": "]") );
        }
    }
    putchar(']');
}
# endif // OUTPUT_RESULT

void randomFill(double* nums, int size) {
    for (int i = 0; i < size; ++i) {
        nums[i] = ((double)rand() / RAND_MAX) * 1e4;
    }
}

__global__ void matrixMultipleKernel(double* A, double* B, double* C, int m, int n, int k) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    int j = blockIdx.y * blockDim.y + threadIdx.y;

    if (i < m && j < k) {
        double sum = 0;
        for (int t = 0; t < n; ++t) {
            int indexA = i * n + t;
            int indexB = t * k + j;
            sum += A[indexA] * B[indexB];
        }
        int indexC = i * k + j;
        C[indexC] = sum;
    }
}

constexpr int blockLen = 16;

int main() {
    double* A, * B, * C;
    int m, n, k;

    scanf("%d %d %d", &m, &n, &k);

    A = (double*)malloc(m * n * sizeof(double));
    B = (double*)malloc(n * k * sizeof(double));
    C = (double*)malloc(m * k * sizeof(double));

    randomFill(A, m * n);
    randomFill(B, n * k);

# ifdef TIME_INFO
    double t1 = wtime();
# endif // TIME_INFO

    double* devA, * devB, * devC;
    cudaMalloc(&devA, m * n * sizeof(double));
    cudaMalloc(&devB, n * k * sizeof(double));
    cudaMalloc(&devC, m * k * sizeof(double));

    cudaMemcpy(devA, A, m * n * sizeof(double), cudaMemcpyHostToDevice);
    cudaMemcpy(devB, B, n * k * sizeof(double), cudaMemcpyHostToDevice);

# ifdef TIME_INFO
    double t2 = wtime();
    printf("cudaMalloc and cudaMemcpy: %lf s.\n", t2 - t1);
# endif // TIME_INFO

    dim3 grid((m + blockLen - 1) / blockLen, (k + blockLen - 1) / blockLen);
    dim3 block(blockLen, blockLen);
    matrixMultipleKernel<<<grid, block>>>(devA, devB, devC, m, n, k);

    cudaDeviceSynchronize();

# ifdef TIME_INFO
    double t3 = wtime();
    printf("matrixMultipleKernel: %lf s.\n", wtime() - t2);
# endif // TIME_INFO

    cudaMemcpy(C, devC, m * k * sizeof(double), cudaMemcpyDeviceToHost);

    cudaFree(devA);
    cudaFree(devB);
    cudaFree(devC);

# ifdef TIME_INFO
    printf("cudaMemcpy and cudaFree: %lf s.\n", wtime() - t3);
    printf("Total: %lf s.\n", wtime() - t1);
# endif // TIME_INFO

# ifdef OUTPUT_RESULT
    matrixPrint(C, m, k);
# endif // OUTPUT_RESULT

    free(A);
    free(B);
    free(C);

    return 0;
}