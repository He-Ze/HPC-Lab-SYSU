#include<stdio.h>
#include<stdlib.h>
#include<pthread.h>
#define isprint 1

double a[2048][2048],b[2048][2048],c[2048][2048];
int m,n,k;
struct v {
	int roll,col;
};

void *thread(void *x) {
	struct v *data = (v*)x; 
	double sum = 0;
	for(int t = 0; t< n; t++){
		sum += a[data->roll][t] * b[t][data->col];
	}
	c[data->roll][data->col] = sum;
	pthread_exit(0);
}

int main(int argc, char * argv[] ){
	clock_t start,end;
	m=atoi(argv[1]);
	n=atoi(argv[2]);
	k=atoi(argv[3]);
		
	srand((unsigned)time(0));
	for (int i = 0; i < m; i++){
		for (int j = 0; j < n; j++){
			a[i][j] = (double)rand() / (double)(RAND_MAX)*1000;
			b[i][j] = (double)rand() / (double)(RAND_MAX)*1000;
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
		for (int y = 0; y < n; y++){
			for (int j = 0; j < k; j++){
				printf("%.2f  \t",b[y][j]);
			}
			printf("\n");
		}
	}
	
	start=clock();
	for(int i = 0; i < m; i++) {
		for(int j = 0; j < k; j++) {
			struct v *data = (struct v *) malloc(sizeof(struct v));
			data->roll = i;
			data->col = j;
			pthread_t t;
			pthread_attr_t attr;
			pthread_attr_init(&attr);
			pthread_create(&t,&attr,thread,data);
			pthread_join(t, NULL);
		}
	}
	end=clock();
	double endtime=(double)(end-start)/CLOCKS_PER_SEC;
	if(isprint){
		printf("=======================================================================\n");
		printf("矩阵C有%d行%d列 ：\n",m,k);
		for (int y = 0; y < m; y++){
			for (int j = 0; j < k; j++){
				printf("%.2f  \t",c[y][j]);
			}
			printf("\n");
		}
	}
	printf("Pthread计算GEMM矩阵乘法已完成，用时：%f ms.\n",endtime*1000);
	return 0;
}