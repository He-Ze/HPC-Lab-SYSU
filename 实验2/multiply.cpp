#include "matrix_multiply.h"
int m,n,k;
double a[2048][2048],b[2048][2048],c[2048][2048];
int main()
{
	clock_t start,end;
	printf("请依次输入m，n，k的值（范围512～2048）：");
	scanf("%d%d%d",&m,&n,&k);
	srand((unsigned)time(0));
	for (int i = 0; i < m; i++){
		for (int j = 0; j < n; j++){
			a[i][j] = (double)rand() / (double)(RAND_MAX)*100;
			b[i][j] = (double)rand() / (double)(RAND_MAX)*100;
		}
	}
	start=clock();
	matrix_multiply((double*)&a,(double*)&b,(double*)&c,m,n,k);
	end=clock();
	double endtime=(double)(end-start)/CLOCKS_PER_SEC;
	printf("Coppersmith-Winograd优化矩阵乘法已完成，用时：%f ms.\n",endtime*1000);
	return 0;
}
