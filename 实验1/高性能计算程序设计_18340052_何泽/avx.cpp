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


void strassen(double* matA, double* matB, double* matC, const int M, const int N, const int K)
{
	if ((M <= 64) || (M%2 != 0 ||N%2 != 0 ||K%2!=0))
		return gemm(matA, matB, matC, M, N, K);
	memset(matC, 0, M*M*sizeof(int));
	int offset = 0;

	//M1 = (A11+A22)*(B11+B22)
	std::vector<double> M1((M / 2) * (N / 2));
	{
		memset(&M1[0], 0, M1.size()*sizeof(double));
		//M1_0 = (A11+A22)
		std::vector<double> M1_0((M / 2) * (K / 2));
		offset = M*M / 2 + K / 2;
		for (int i = 0; i < M / 2; i++){
			for (int j = 0; j < K/2; j++){
				const int baseIdx = i*M + j;
				M1_0[i*K/2+j] = matA[baseIdx] + matA[baseIdx + offset];
			}
		}
		//M1_1 = (B11+B22)
		std::vector<double> M1_1((K / 2) * (N / 2));
		offset = K*M / 2 + N / 2;
		for (int i = 0; i < K / 2; i++){
			for (int j = 0; j < N / 2; j++){
				const int baseIdx = i*M + j;
				M1_1[i*N/2+j] = matB[baseIdx] + matB[baseIdx + offset];
			}
		}
		strassen(&M1_0[0], &M1_1[0], &M1[0], M / 2, N / 2, K / 2);
	}

	//M2 = (A21+A22)*B11
	std::vector<double> M2((M / 2) * (N / 2));
	{
		memset(&M2[0], 0, M2.size()*sizeof(double));
		//M2_0 = (A21+A22)
		std::vector<double> M2_0((M / 2) * (K / 2));
		offset = K / 2;
		for (int i = M / 2; i < M; i++){
			for (int j = 0; j < K / 2; j++){
				const int baseIdx = i*M + j;
				M2_0[(i-M/2)*K/2+j] = matA[baseIdx] + matA[baseIdx + offset];
			}
		}
		//M2_2 = B11
		strassen(&M2_0[0], &matB[N / 2], &M2[0], M / 2, N / 2, K / 2);
	}

	//M3 = A11*(B12-B22)
	std::vector<double> M3((M / 2) * (N / 2));
	{
		memset(&M3[0], 0, M3.size()*sizeof(double));
		//M3_0 = A11
		//M3_1 = (B12-B22)
		std::vector<double> M3_1((K / 2) * (N / 2));
		offset = K*M / 2;
		for (int i = 0; i < K/2; i++){
			for (int j = N/2; j < N; j++){
				const int baseIdx = i*M + j;
				M3_1[i*N/2+j-N/2] = matB[baseIdx] - matB[baseIdx + offset];
			}
		}
		strassen(matA, &M3_1[0], &M3[0], M / 2, N / 2, K / 2);
	}

	//M4 = A22*(B21-B11)
	std::vector<double> M4((M / 2) * (N / 2));
	{
		memset(&M4[0], 0, M4.size()*sizeof(double));
		//M4_0 = A22
		//M4_1 = (B12-B22)
		std::vector<double> M4_1((K / 2) * (N / 2));
		offset = K*M / 2;
		for (int i = 0; i < K / 2; i++){
			for (int j = N / 2; j < N; j++){
				const int baseIdx = i*M + j;
				M4_1[i*N/2+j-N/2] = matB[baseIdx + offset] - matB[baseIdx];
			}
		}
		strassen(matA + M*M / 2 + K / 2, &M4_1[0], &M4[0], M / 2, K / 2, N / 2);
	}

	//M5 = (A11+A12)*B22
	std::vector<double> M5((M / 2) * (N / 2));
	{
		memset(&M5[0], 0, M5.size()*sizeof(double));
		//M5_0 = (A11+A12)
		std::vector<double> M5_0((M / 2) * (K / 2));
		offset = K / 2;
		for (int i = 0; i < M/2; i++){
			for (int j = 0; j < K / 2; j++){
				const int baseIdx = i*M + j;
				M5_0[i*K / 2 + j] = matA[baseIdx] + matA[baseIdx + offset];
			}
		}
		//M5_1 = B22
		strassen(&M5_0[0], &matB[K*M / 2 + N / 2], &M5[0], M / 2, N / 2, K / 2);
	}

	//M6 = (A21-A11)*(B11+B12)
	std::vector<double> M6((M / 2) * (N / 2));
	{
		memset(&M6[0], 0, M6.size()*sizeof(double));
		//M6_0 = (A21-A11)
		std::vector<double> M6_0((M / 2) * (K / 2));
		offset = K*N / 2;
		for (int i = 0; i < M / 2; i++){
			for (int j = 0; j < K/2; j++){
				const int baseIdx = i*M + j;
				M6_0[i*K/2+j] = matA[baseIdx + offset] - matA[baseIdx];
			}
		}
		//M6_1 = (B11+B12)
		std::vector<double> M6_1((K / 2) * (N / 2));
		offset = N / 2;
		for (int i = 0; i < K / 2; i++){
			for (int j = 0; j < N/2; j++){
				const int baseIdx = i*M + j;
				M6_1[i*N/2+j] = matB[baseIdx] + matB[baseIdx + offset];
			}
		}
		strassen(&M6_0[0], &M6_1[0], &M6[0], M / 2, N / 2, K / 2);
	}

	//M7 = (A12-A22)*(B21+B22)
	std::vector<double> M7((M / 2) * (N / 2));
	{
		memset(&M7[0], 0, M7.size()*sizeof(double));
		//M7_0 = (A12-A22)
		std::vector<double> M7_0((M / 2) * (K / 2));
		offset = M*M / 2;
		for (int i = 0; i < M / 2; i++){
			for (int j = K/2; j < K; j++){
				const int baseIdx = i*M + j;
				M7_0[i*K / 2 + j - K / 2] = matA[baseIdx] - matA[baseIdx + offset];
			}
		}
		//M7_1 = (B21+B22)
		std::vector<double> M7_1((K / 2) * (N / 2));
		offset = N / 2;
		for (int i = K/2; i < K; i++){
			for (int j = 0; j < N / 2; j++){
				const int baseIdx = i*M + j;
				M7_1[(i-K/2)*N / 2 + j] = matB[baseIdx] + matB[baseIdx + offset];
			}
		}
		strassen(&M7_0[0], &M7_1[0], &M7[0], M / 2, N / 2, K / 2);
	}
	for (int i = 0; i < M / 2;i++){
		for (int j = 0; j < N / 2;j++){
			const int idx = i*N / 2 + j;
			//C11 = M1+M4-M5+M7
			matC[i*M + j] = M1[idx] + M4[idx] - M5[idx] + M7[idx];
			//C12 = M3+M5
			matC[i*M + j + N/2] = M3[idx] + M5[idx];
			//C21 = M2+M4
			matC[(i+M/2)*M + j] = M2[idx] + M4[idx];
			//C22 = M1-M2+M3+M6
			matC[(i+M/2)*M + j + N/2] = M1[idx] - M2[idx] + M3[idx] + M6[idx];
		}
	}
}


void winograd(double* matA, double* matB, double* matC, const int M, const int N, const int K)
{
	if ((M <= 64) || (M % 2 != 0 || N % 2 != 0 || K % 2 != 0))
		return gemm(matA, matB, matC, M, N, K);
	memset(matC, 0, M*M*sizeof(double));
	int offset = 0;

	std::vector<double> S1((M / 2) * (K / 2));
	std::vector<double> S2((M / 2) * (K / 2));
	std::vector<double> S3((M / 2) * (K / 2));
	std::vector<double> S4((M / 2) * (K / 2));
	for (int i = 0; i < M / 2;i++){
		for (int j = 0; j < K / 2;j++){
			const double idx = i*K / 2 + j;
			//S1 = A21 + A22
			S1[idx] = matA[(i + M / 2)*M + j] + matA[(i + M / 2)*M + j + K / 2];
			//S2 = S1 - A11
			S2[idx] = S1[idx] - matA[i*M + j];
			//S3 = A11 - A21
			S3[idx] = matA[i*M + j] - matA[(i + M / 2)*M + j];
			//S4 = A12 - S2
			S4[idx] = matA[i*M + j + K / 2] - S2[idx];
		}
	}
	std::vector<double> T1((K / 2) * (N / 2));
	std::vector<double> T2((K / 2) * (N / 2));
	std::vector<double> T3((K / 2) * (N / 2));
	std::vector<double> T4((K / 2) * (N / 2));
	for (int i = 0; i < K / 2; i++){
		for (int j = 0; j < N / 2; j++){
			const double idx = i*N / 2 + j;
			//T1 = B21 - B11
			T1[idx] = matB[(i + K / 2)*M + j] - matB[i*M + j];
			//T2 = B22 - T1
			T2[idx] = matB[(i + K / 2)*M + j + N / 2] - T1[idx];
			//T3 = B22 - B12
			T3[idx] = matB[(i + K / 2)*M + j + N / 2] - matB[i*M + j + N / 2];
			//T4 = T2 - B21
			T4[idx] = T2[idx] - matB[(i + K / 2)*M + j];
		}
	}

	//M1 = A11*B11
	std::vector<double> M1((M / 2) * (N / 2));
	{
		memset(&M1[0], 0, M1.size()*sizeof(double));
		winograd(matA, matB, &M1[0], M / 2, N / 2, K / 2);
	}

	//M2 = A12*B21
	std::vector<double> M2((M / 2) * (N / 2));
	{
		memset(&M2[0], 0, M2.size()*sizeof(double));
		winograd(matA + K / 2, matB + K*M/2, &M2[0], M / 2, N / 2, K / 2);
	}

	//M3 = S4*B22
	std::vector<double> M3((M / 2) * (N / 2));
	{
		memset(&M3[0], 0, M3.size()*sizeof(double));
		winograd(&S4[0], matB + K*M/2 + N / 2, &M3[0], M / 2, N / 2, K / 2);
	}

	//M4 = A22*T4
	std::vector<double> M4((M / 2) * (N / 2));
	{
		memset(&M4[0], 0, M4.size()*sizeof(double));
		winograd(matA + M*M / 2 + K / 2, &T4[0], &M4[0], M / 2, N / 2, K / 2);
	}

	//M5 = S1*T1
	std::vector<double> M5((M / 2) * (N / 2));
	{
		memset(&M5[0], 0, M5.size()*sizeof(double));
		winograd(&S1[0], &T1[0], &M5[0], M / 2, N / 2, K / 2);
	}

	//M6 = S2*T2
	std::vector<double> M6((M / 2) * (N / 2));
	{
		memset(&M6[0], 0, M6.size()*sizeof(double));
		winograd(&S2[0], &T2[0], &M6[0], M / 2, N / 2, K / 2);
	}

	//M7 = S3*T3
	std::vector<double> M7((M / 2) * (N / 2));
	{
		memset(&M7[0], 0, M7.size()*sizeof(double));
		winograd(&S3[0], &T3[0], &M7[0], M / 2, N / 2, K / 2);
	}

	for (int i = 0; i < M / 2; i++){
		for (int j = 0; j < N / 2; j++){
			const double idx = i*N / 2 + j;
			//U1 = M1 + M2
			const auto U1 = M1[idx] + M2[idx];
			//U2 = M1 + M6
			const auto U2 = M1[idx] + M6[idx];
			//U3 = U2 + M7
			const auto U3 = U2 + M7[idx];
			//U4 = U2 + M5
			const auto U4 = U2 + M5[idx];
			//U5 = U4 + M3
			const auto U5 = U4 + M3[idx];
			//U6 = U3 - M4
			const auto U6 = U3 - M4[idx];
			//U7 = U3 + M5
			const auto U7 = U3 + M5[idx];

			//C11 = U1
			matC[i*M + j] = U1;
			//C12 = U5
			matC[i*M + j + N / 2] = U5;
			//C21 = U6
			matC[(i + M / 2)*M + j] = U6;
			//C22 = U7
			matC[(i + M / 2)*M + j + N / 2] = U7;
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
	
	start=clock();
	strassen((double*)&a,(double*)&b,(double*)&c,m,n,k);
	end=clock();
	endtime=(double)(end-start)/CLOCKS_PER_SEC;
	if(isprint){
		printf("=======================================================================\n");
		printf("矩阵D有%d行%d列 ：\n",m,k);
		for (int y = 0; y < m; y++){
			for (int j = 0; j < n; j++){
				printf("%.2f  \t",c[y][j]);
			}
			printf("\n");
		}
	}
	printf("Strassen优化矩阵乘法已完成，用时：%f ms.\n",endtime*1000);

	start=clock();
	winograd((double*)&a,(double*)&b,(double*)&c,m,n,k);
	end=clock();
	endtime=(double)(end-start)/CLOCKS_PER_SEC;
	if(isprint){
		printf("=======================================================================\n");
		printf("矩阵E有%d行%d列 ：\n",m,k);
		for (int y = 0; y < m; y++){
			for (int j = 0; j < n; j++){
				printf("%.2f  \t",c[y][j]);
				c[y][j]=0;
			}
			printf("\n");
		}
	}
	printf("Coppersmith-Winograd优化矩阵乘法已完成，用时：%f ms.\n",endtime*1000);

	__m256d vec_res = _mm256_setzero_pd();
	__m256d vec_1 = _mm256_setzero_pd();
	__m256d vec_2 = _mm256_setzero_pd();

	start=clock();
	for (int i = 0; i < m; i++){
		for (int j = 0; j < m; j++){
			vec_1 = _mm256_set1_pd(a[i][j]);
			for (int k = 0; k < m; k += 4){
				vec_2 = _mm256_load_pd(&b[j][k]);
				vec_res = _mm256_load_pd(&c[i][k]);
				vec_res = _mm256_add_pd(vec_res ,_mm256_mul_pd(vec_1, vec_2));
				_mm256_store_pd(&c[i][k], vec_res);
			}
		}
	}
	end=clock();
	endtime=(double)(end-start)/CLOCKS_PER_SEC;
	
	if(isprint){
		printf("=======================================================================\n");
		printf("矩阵F有%d行%d列 ：\n",m,k);
		for (int y = 0; y < m; y++){
			for (int j = 0; j < n; j++){
				printf("%.2f  \t",c[y][j]);
			}
			printf("\n");
		}
	}
	printf("AVX优化矩阵乘法已完成，用时：%f ms.\n",endtime*1000);
	return 0;
}
