#include<stdio.h>
#include<stdlib.h>
#include<mpi.h>
#include<iostream>
#define isprint 0

int main(int argc, char * argv[] ){
	double start, stop;
	double *a, *b, *c, *buffer, *ans;
	int m,n,k;
	m=atoi(argv[1]);
	n=atoi(argv[2]);
	k=atoi(argv[3]);
	int rank, numprocs, line;
	
	MPI_Init(NULL,NULL);
	MPI_Comm_rank(MPI_COMM_WORLD, &rank);
	MPI_Comm_size(MPI_COMM_WORLD, &numprocs);
	line = m/numprocs;
	b = new double [ n * k ];
	ans = new double [ m * k ];

	if( rank ==0 ){
		a = new double [ m * n ];
		c = new double [ m * k ];

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
		for(int i=1;i<numprocs;i++){
			MPI_Send( b, n*k, MPI_DOUBLE, i, 0, MPI_COMM_WORLD );
		}
		for(int i=1;i<numprocs;i++){
			MPI_Send( a + (i-1)*line*n, n*line, MPI_DOUBLE, i, 1, MPI_COMM_WORLD);
		}
		for(int i = (numprocs-1)*line;i<m;i++){
			for(int j=0;j<n;j++){
				double temp = 0;
				for(int t=0;t<n;t++)
					temp += a[i*n+t]*b[t*k+j];
				c[i*k+j] = temp;
			}
		}
		for(int t=1;t<numprocs;t++){
			MPI_Recv( ans, line*k, MPI_DOUBLE, t, 3, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
			for(int i=0;i<line;i++){
				for(int j=0;j<k;j++){
					c[ ((t-1)*line + i)*k + j] = ans[i*k+j];
				}
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
		delete [] a;
	}
	else{
		buffer = new double [ n * line ];

		MPI_Recv(b, n*k, MPI_DOUBLE, 0, 0, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
		MPI_Recv(buffer, n*line, MPI_DOUBLE, 0, 1, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
		for(int i=0;i<line;i++){
			for(int j=0;j<n;j++){
				double temp=0;
				for(int t=0;t<n;t++)
					temp += buffer[i*n+t]*b[t*k+j];
				ans[i*k+j] = temp;
			}
		}
		MPI_Send(ans, line*k, MPI_DOUBLE, 0, 3, MPI_COMM_WORLD);
		delete [] buffer;
		delete [] ans;
	}
	delete [] b;
	MPI_Finalize();
	return 0;
}
