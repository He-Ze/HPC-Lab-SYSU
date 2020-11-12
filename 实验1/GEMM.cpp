#include <stdio.h>
#include <stdlib.h>
#include <time.h>
#include <math.h>
#include <vector>
#include <string.h>
#include <immintrin.h>
#include <emmintrin.h>
#define isprint 0

int m,n,k;
double a[2048][2048],b[2048][2048],c[2048][2048];

void gemm(double* matA,double* matB,double* matC,const int M,const int N,const int K)
{
	for (int i = 0; i < M;i++){
		for (int j = 0; j < N;j++){
			double sum = 0;
			for (int k = 0; k < K;k++)
				sum += matA[i*M + k] * matB[k*N + j];
			matC[i*K + j] = sum;
		}
	}
}

int main()
{
	clock_t start,end;
	
	
	printf("请依次输入m，n，k的值（范围512～2048）：");
	scanf("%d%d%d",&m,&n,&k);
	if(m>2048||n>2048||k>2048){
		printf("请输入小于2048的值");
		scanf("%d%d%d",&m,&n,&k);
	}
	//double a[m][m],b[m][m],c[m][m];
	srand((unsigned)time(0));
	for (int i = 0; i < m; i++){
		for (int j = 0; j < n; j++){
			a[i][j] = (double)rand() / (double)(RAND_MAX)*100;
			b[i][j] = (double)rand() / (double)(RAND_MAX)*100;
		}
	}

	if(isprint){
		printf("=======================================================================\n");
		printf("矩阵A有%d行%d列 ：\n",m,n);
		for (int y = 0; y < m; y++){
			for (int j = 0; j < n; j++){
				printf("%.2f  \t",a[y][j]);
			}
			printf("\n");
		}

		printf("矩阵B有%d行%d列 ：\n",n,k);
		for (int y = 0; y < m; y++){
			for (int j = 0; j < n; j++){
				printf("%.2f  \t",b[y][j]);
			}
			printf("\n");
		}
	}
	
	start=clock();
	gemm((double*)&a,(double*)&b,(double*)&c,m,n,k);
	
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
	printf("GEMM通用矩阵乘法已完成，用时：%f ms.\n",endtime*1000);

	return 0;
}
