#include<stdio.h>
#include<stdlib.h>
#include<mpi.h>
#include<iostream>
#define isprint 0

int main(int argc, char * argv[] ){
	double start, stop;
	int m,n,k;
	m=atoi(argv[1]);
	n=atoi(argv[2]);
	k=atoi(argv[3]);
	int rank,numprocs,line;

	MPI_Init(NULL,NULL);
	MPI_Comm_rank(MPI_COMM_WORLD, &rank);
	MPI_Comm_size(MPI_COMM_WORLD, &numprocs);
	
	line = m / numprocs;
	double * a = new double [m*n];
	double * b = new double [n*k];
	double* c = new double [m*k];
	double * local_a = new double [line*n];
	double * ans = new double [m*k];

	if( rank==0 ){
		srand((unsigned)time(0));
		for(int i=0;i<m; i++){
			for(int j=0;j<n; j++){
				a[i*n+j] = (double)rand()/ (double)(RAND_MAX)*1000;
			}
		}
		for(int i=0;i<n; i++){
			for(int j=0;j<k; j++){
				b[i*k+j] = (double)rand()/ (double)(RAND_MAX)*1000;
			}
		}
		if(isprint){
			printf("矩阵a：\n");
			for(int i=0;i<m;i++){
				for(int j=0;j<n;j++)
					printf("%.2f \t",a[i*n+j]);
				printf("\n");
			}
			printf("矩阵b：\n");
			for(int i=0;i<n;i++){
				for(int j=0;j<k;j++)
					printf("%.2f \t",b[i*k+j]);
				printf("\n");
			}
		}
		start = MPI_Wtime();
		MPI_Scatter(a, line*n, MPI_DOUBLE, local_a, line*n, MPI_DOUBLE, 0, MPI_COMM_WORLD );
		MPI_Bcast(b, n*k, MPI_DOUBLE, 0, MPI_COMM_WORLD);
		for(int i= 0; i< m;i++){
			for(int j=0;j<n;j++){
				double temp = 0;
				for(int t=0;t<n;t++)
					temp += a[i*n+t] * b[t*k + j];
				ans[i*k + j ] = temp;
			}
		}
		MPI_Gather( ans, line*k, MPI_DOUBLE, c, line*k, MPI_DOUBLE, 0, MPI_COMM_WORLD );
		for(int i = (numprocs-1)*line;i<m;i++){
			for(int j=0;j<n;j++){
				double temp = 0;
				for(int t=0;t<n;t++)
					temp += a[i*n+t]*b[t*k+j];
				c[i*k+j] = temp;
			}
		}
		stop = MPI_Wtime();
		if(isprint){
			printf("矩阵c：\n");
			for(int i=0;i<m;i++){
				for(int j=0;j<k;j++)
					printf("%.2f \t",c[i*k+j]);
				printf("\n");
			}
		}
		printf("计算所用时间：%lfs\n",stop-start);
	}
	else{
		double * buffer = new double [ n * line ];
		MPI_Scatter(a, line*n, MPI_DOUBLE, buffer, line*n, MPI_DOUBLE, 0, MPI_COMM_WORLD );
		MPI_Bcast( b, n * k, MPI_DOUBLE, 0, MPI_COMM_WORLD );
		for(int i=0;i<line;i++){
			for(int j=0;j<n;j++){
				double temp=0;
				for(int t=0;t<n;t++)
					temp += buffer[i*n+t]*b[t*k+j];
				ans[i*k+j] = temp;
			}
		}
		MPI_Gather(ans, line*k, MPI_DOUBLE, c, line*k, MPI_DOUBLE, 0, MPI_COMM_WORLD );
		delete [] buffer;
	}
	delete [] a;
	MPI_Finalize();
	return 0;
}
