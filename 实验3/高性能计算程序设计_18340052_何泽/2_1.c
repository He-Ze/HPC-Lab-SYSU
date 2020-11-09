#include <stdio.h>
#include <stdlib.h>
#include <pthread.h>
 
#define isprint 0
int ARRAY_SIZE;
int NUM_THREADS;
int array[10000];
int global_index = 0;
int global_sum = 0;
pthread_mutex_t mutex=PTHREAD_MUTEX_INITIALIZER;

void* add ()
{
	pthread_mutex_lock(&mutex);
	global_sum += array[global_index];
	global_index++;
	pthread_mutex_unlock(&mutex);
}

int main (int argc, char * argv[])
{
	clock_t start,end;
	ARRAY_SIZE=atoi(argv[1]);
	NUM_THREADS=atoi(argv[2]);
	pthread_t thread[NUM_THREADS];
	pthread_mutex_init (&mutex, NULL);
	srand((unsigned)time(0));
	for (int i=0;i<ARRAY_SIZE;i++)
		array[i]=(int)rand()%100;
	if(isprint){
		printf("数组元素：");
		for (int i=0;i<ARRAY_SIZE;i++)
			printf("%d ",array[i]);
		printf("\n");
	}
	
	int sum=0;
	start=clock();
	for (int i=0;i<ARRAY_SIZE;i++) {
		sum+=array[i];
	}
	end=clock();
	double endtime1=(double)(end-start)/CLOCKS_PER_SEC;
	printf("这%d个数的和为%d，直接计算用时%fms\n", ARRAY_SIZE, sum,endtime1*1000);
	start=clock();
	for (int i=0;i<NUM_THREADS;i++){
		for (int t=0;t<ARRAY_SIZE/NUM_THREADS;t++) 
			pthread_create(&thread[i], NULL, add, NULL);
	}
	for (int i=0;i<NUM_THREADS;i++){
		for (int t=0;t<ARRAY_SIZE/NUM_THREADS;t++) 
			pthread_join(thread[i], NULL);
	}
	end=clock();
	double endtime2=(double)(end-start)/CLOCKS_PER_SEC;
	pthread_mutex_destroy(&mutex);
	printf("这%d个数的和为%d，Pthread用时%fms\n", ARRAY_SIZE, global_sum,endtime2*1000);
	return 0;
}