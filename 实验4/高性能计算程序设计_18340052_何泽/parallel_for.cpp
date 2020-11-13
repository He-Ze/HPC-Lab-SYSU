#include"parallel_for.h"
void parallel_for(int start, int end, int increment, void *(*functor)(void *), void *arg, int num_threads){
    pthread_t *threads = (pthread_t *)malloc(num_threads * sizeof(pthread_t));
    for_index *a = (for_index *)malloc(num_threads * sizeof(for_index));
    int block = (end - start) / num_threads;
    for (int thread = 0; thread < num_threads; thread++){
        a[thread].args = arg;
        a[thread].start = start + thread * block;
        a[thread].end = a[thread].start + block;
        if (thread == (num_threads - 1))
            a[thread].end = end;
        a[thread].increment = increment;
        pthread_create(&threads[thread], NULL, functor, (void *)(a + thread));
    }
    for (int thread = 0; thread < num_threads; thread++)
        pthread_join(threads[thread], NULL);
    free(threads);
    free(a);
}