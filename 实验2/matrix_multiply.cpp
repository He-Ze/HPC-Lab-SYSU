#include <stdio.h>
#include <stdlib.h>
#include <time.h>
#include <math.h>
#include <vector>
#include <string.h>

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

void matrix_multiply(double* matA, double* matB, double* matC, const int M, const int N, const int K)
{
	if ((M <= 64) || (M % 2 != 0 || N % 2 != 0 || K % 2 != 0)){
		return gemm(matA, matB, matC, M, N, K);
	}
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
		matrix_multiply(matA, matB, &M1[0], M / 2, N / 2, K / 2);
	}

	//M2 = A12*B21
	std::vector<double> M2((M / 2) * (N / 2));
	{
		memset(&M2[0], 0, M2.size()*sizeof(double));
		matrix_multiply(matA + K / 2, matB + K*M/2, &M2[0], M / 2, N / 2, K / 2);
	}

	//M3 = S4*B22
	std::vector<double> M3((M / 2) * (N / 2));
	{
		memset(&M3[0], 0, M3.size()*sizeof(double));
		matrix_multiply(&S4[0], matB + K*M/2 + N / 2, &M3[0], M / 2, N / 2, K / 2);
	}

	//M4 = A22*T4
	std::vector<double> M4((M / 2) * (N / 2));
	{
		memset(&M4[0], 0, M4.size()*sizeof(double));
		matrix_multiply(matA + M*M / 2 + K / 2, &T4[0], &M4[0], M / 2, N / 2, K / 2);
	}

	//M5 = S1*T1
	std::vector<double> M5((M / 2) * (N / 2));
	{
		memset(&M5[0], 0, M5.size()*sizeof(double));
		matrix_multiply(&S1[0], &T1[0], &M5[0], M / 2, N / 2, K / 2);
	}

	//M6 = S2*T2
	std::vector<double> M6((M / 2) * (N / 2));
	{
		memset(&M6[0], 0, M6.size()*sizeof(double));
		matrix_multiply(&S2[0], &T2[0], &M6[0], M / 2, N / 2, K / 2);
	}

	//M7 = S3*T3
	std::vector<double> M7((M / 2) * (N / 2));
	{
		memset(&M7[0], 0, M7.size()*sizeof(double));
		matrix_multiply(&S3[0], &T3[0], &M7[0], M / 2, N / 2, K / 2);
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
