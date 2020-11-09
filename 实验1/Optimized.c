# include <stdio.h>
# include <stdlib.h>
# include <immintrin.h>
# include <sys/time.h>
//# define VERIFY

typedef struct Matrix {
    double* nums;
    int m;
    int n;
} Matrix;

void matrixInit(Matrix* mat, int m, int n) {
    mat->nums = (double*)malloc(n * m * sizeof(double));
    mat->m = m;
    mat->n = n;
}

void matrixRandomFill(const Matrix mat) {
    for (int i = 0; i < mat.m; ++i) {
        for (int j = 0; j < mat.n; ++j) {
            /*register*/ int index = i * mat.n + j;
            mat.nums[index] = ((double)rand() / RAND_MAX) * 1e4;
        }
    }
}

# define I_STRIDE 64
# define J_STRIDE 64

void matrixMultiple(const Matrix A, const Matrix B, const Matrix C) {
    for (int i = 0; i <= C.m - I_STRIDE; i += I_STRIDE) {
        for (int j = 0; j <= C.n - J_STRIDE; j += J_STRIDE) {
            __m256d res[I_STRIDE][J_STRIDE/4];
            # pragma unroll(I_STRIDE)
            for (int i1 = 0; i1 < I_STRIDE; ++i1) {
                # pragma unroll(J_STRIDE/4)
                for (int i2 = 0; i2 < J_STRIDE/4; ++i2) {
                    res[i1][i2] = _mm256_set1_pd(0);
                }
            }
            for (int k = 0; k < A.n; ++k) {
                __m256d other_k_j_vt4[J_STRIDE/4];
                # pragma unroll(J_STRIDE/4)
                for (int i2 = 0; i2 < J_STRIDE/4; ++i2) {
                    other_k_j_vt4[i2] = _mm256_loadu_pd(B.nums + k * B.n + (j + 4*i2));
                }
                # pragma unroll(I_STRIDE)
                for (int i1 = 0; i1 < I_STRIDE; ++i1) {
                    __m256d this_i_k = _mm256_set1_pd(A.nums[(i + i1) * A.n + k]);
                    # pragma unroll(J_STRIDE/4)
                    for (int i2 = 0; i2 < J_STRIDE/4; ++i2) {
                        res[i1][i2] = _mm256_add_pd(res[i1][i2], _mm256_mul_pd(this_i_k, other_k_j_vt4[i2]));
                    }
                }
            }
            # pragma unroll(I_STRIDE)
            for (int i1 = 0; i1 < I_STRIDE; ++i1) {
                # pragma unroll(J_STRIDE/4)
                for (int i2 = 0; i2 < J_STRIDE/4; ++i2) {
                    _mm256_storeu_pd(C.nums + (i + i1) * C.n + (j + 4*i2), res[i1][i2]);
                }
            }
        }
    }
    int rest_m = C.m % I_STRIDE;
    int rest_n = C.n % J_STRIDE;
    for (int i = 0; i < C.m - rest_m; ++i) {
        for (int j = C.n - rest_n; j <= C.n - 4; j += 4) {
            __m256d res = _mm256_set1_pd(0);
            for (int k = 0; k < A.n; ++k) {
                __m256d other_k_j_vt4 = _mm256_loadu_pd(B.nums + k * B.n + j);
                __m256d this_i_k = _mm256_set1_pd(A.nums[i * A.n + k]);
                res = _mm256_add_pd(res, _mm256_mul_pd(this_i_k, other_k_j_vt4));
            }
            _mm256_storeu_pd(C.nums + i * C.n + j, res);
        }
        for (int j = C.n - (C.n % 4); j < C.n; ++j) {
            C.nums[i * C.n + j] = 0;
            for (int k = 0; k < A.n; ++k) {
                C.nums[i * C.n + j] += A.nums[i * A.n + k] * B.nums[k * B.n + j];
            }
        }
    }
    for (int i = C.m - rest_m; i < C.m; ++i) {
        for (int j = 0; j <= C.n - 4; j += 4) {
            __m256d res = _mm256_set1_pd(0);
            for (int k = 0; k < A.n; ++k) {
                __m256d other_k_j_vt4 = _mm256_loadu_pd(B.nums + k * B.n + j);
                __m256d this_i_k = _mm256_set1_pd(A.nums[i * A.n + k]);
                res = _mm256_add_pd(res, _mm256_mul_pd(this_i_k, other_k_j_vt4));
            }
            _mm256_storeu_pd(C.nums + i * C.n + j, res);
        }
        for (int j = C.n - (C.n % 4); j < C.n; ++j) {
            C.nums[i * C.n + j] = 0;
            for (int k = 0; k < A.n; ++k) {
                C.nums[i * C.n + j] += A.nums[i * A.n + k] * B.nums[k * B.n + j];
            }
        }
    }
}

void matrixPrint(const Matrix mat) {
    putchar('[');
    for (int i = 0; i < mat.m; ++i) {
        putchar('[');
        for (int j = 0; j < mat.n; ++j) {
            /*register*/ int index = i * mat.n + j;
            printf("%g", mat.nums[index]);
            printf(j != mat.n - 1? ", ": (i != mat.m - 1? "],\n": "]") );
        }
    }
    putchar(']');
}

int main() {
    int m, n, k;
    scanf("%d %d %d", &m, &n, &k);

    Matrix A, B, C;
    matrixInit(&A, m, n);
    matrixInit(&B, n, k);
    matrixInit(&C, m, k);
    matrixRandomFill(A);
    matrixRandomFill(B);

    struct timeval start, end;
    gettimeofday(&start, NULL);

    matrixMultiple(A, B, C);

    gettimeofday(&end, NULL);
#ifndef VERIFY
    long long elapsed = (end.tv_sec - start.tv_sec) * 1000000LL + (end.tv_usec - start.tv_usec);
    printf("Elapsed: %g s.\n", elapsed / 1e6);
#else
    matrixPrint(C);
#endif

    return 0;
}
