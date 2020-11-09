#include <stdio.h>
#include <stdlib.h>
#include <pthread.h>
#include <time.h>

long global_num = 0;
long thread_num;
pthread_mutex_t mutex = PTHREAD_MUTEX_INITIALIZER;

void *thread() 
{
	long local_num = 0;
	unsigned int a = rand();
	for (long i = 0; i < thread_num; i++) {
		double x = rand_r(&a) / ((double)RAND_MAX + 1)*2.0-1.0;
		double y = rand_r(&a) / ((double)RAND_MAX + 1)*2.0-1.0;
		if (y>x*x)
			local_num++;
	}
	pthread_mutex_lock(&mutex);
	global_num += local_num;
	pthread_mutex_unlock(&mutex);
}

int main()
{
	long totalpoints = 1000000000;
	int thread_count = 10000;
	thread_num = totalpoints/thread_count;

	srand((unsigned)time(NULL));
	pthread_t *threads = malloc(thread_count * sizeof(pthread_t));
	pthread_attr_t attr;
	pthread_attr_init(&attr);
	for (int i = 0; i < thread_count; i++) {
		pthread_create(&threads[i], &attr, thread, (void *) NULL);
	}
	for (int i = 0; i < thread_count; i++) {
		pthread_join(threads[i], NULL);
	}
	pthread_mutex_destroy(&mutex);
	free(threads);
	printf("x在0～1时y=x^2与x轴之间区域的面积是%f\n", (double)global_num/(double)totalpoints);
	return 0;
}