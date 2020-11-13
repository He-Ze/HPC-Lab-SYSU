#include "parallel_for.h"
#define isprint 1

struct args{
    int *A,*B,*C,*m,*n,*k;
    args(int *tA, int *tB, int *tC, int *tm, int *tn, int *tk){
        A = tA;
        B = tB;
        C = tC;
        m = tm;
        n = tn;
        k = tk;
    }
};

int thread_count=4;
void *gemm(void *arg);
void PrintMatrix(int *A, int *B, int *C, int m, int n, int k);

int main(int argc, char *argv[])
{
    int *A, *B, *C;
    int m, n, k;
    clock_t start,end;
    printf("请依次输入m，n，k的值（范围512～2048）：");
    scanf("%d%d%d",&m,&n,&k);
    if(m>2048||n>2048||k>2048){
        printf("请输入小于2048的值");
        scanf("%d%d%d",&m,&n,&k);
    }
    A = new int[m * n];
    B = new int[n * k];
    C = new int[m * k];
    srand((unsigned)time(0));
    for (int i = 0; i < m; i++){
        for (int j = 0; j < n; j++){
            a[i*m+j] = (double)rand() / (double)(RAND_MAX)*100;
            b[i*n+j] = (double)rand() / (double)(RAND_MAX)*100;
        }
    }
    if(isprint){
        printf("=======================================================================\n");
        printf("矩阵A有%d行%d列 ：\n",m,n);
        for (int y = 0; y < m; y++){
            for (int j = 0; j < n; j++){
                printf("%.2f  \t",a[y*m+j]);
            }
            printf("\n");
        }
        printf("矩阵B有%d行%d列 ：\n",n,k);
        for (int y = 0; y < n; y++){
            for (int j = 0; j < k; j++){
                printf("%.2f  \t",b[y*n+j]);
            }
            printf("\n");
        }
    }
    struct args *arg = new args(A, B, C, &m, &n, &k);
    start=clock();
    parallel_for(0, m, 1, gemm, arg, thread_count);
    end=clock();
    double endtime=(double)(end-start)/CLOCKS_PER_SEC;
    if(isprint){
            printf("=======================================================================\n");
            printf("矩阵C有%d行%d列 ：\n",m,k);
            for (int y = 0; y < m; y++){
                for (int j = 0; j < n; j++){
                    printf("%.2f  \t",c[y][j]);
                }
                printf("\n");
            }
        }
    printf("Parallel_for通用矩阵乘法已完成，用时：%f ms.\n",endtime*1000);
    return 0;
}

void *gemm(void *arg){
    struct for_index *index = (struct for_index *)arg;
    struct args *true_arg = (struct args *)(index->args);
    for (int i = index->start; i < index->end; i = i + index->increment){
        for (int j = 0; j < *true_arg->k; ++j){
            int temp = 0;
            for (int z = 0; z < *true_arg->n; ++z)
                temp += true_arg->A[i * (*true_arg->n) + z] * true_arg->B[z * (*true_arg->k) + j];
            true_arg->C[i * (*true_arg->k) + j] = temp;
        }
    }
    return NULL;
}