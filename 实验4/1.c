#include <stdio.h>
#include <stdlib.h>
#include <time.h>
#include <math.h>
#include <string.h>
#include <omp.h>

#define isprint 0

int m,n,k;
double a[2048][2048],b[2048][2048],c[2048][2048];

int main()
{
	clock_t start,end;
	printf("请依次输入m，n，k的值（范围512～2048）：");
	scanf("%d%d%d",&m,&n,&k);
	if(m>2048||n>2048||k>2048){
		printf("请输入小于2048的值");
		scanf("%d%d%d",&m,&n,&k);
	}
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
	for (int i = 0; i < m;i++){
#pragma omp privite(sum,w) parallel for num_threads(4)
		for (int j = 0; j < n;j++){
			double sum = 0;
			for (int w = 0; w < k;w++)
				sum += a[i][w] * b[w][j];
			c[i][j] = sum;
		}
	}
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
	printf("OpenMP通用矩阵乘法已完成，用时：%f ms.\n",endtime*1000);
	return 0;
}
