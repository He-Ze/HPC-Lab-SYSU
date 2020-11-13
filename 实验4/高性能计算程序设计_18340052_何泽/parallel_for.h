#include <stdio.h>
#include <stdlib.h>
#include <time.h>
#include <math.h>
#include <string.h>
#include <pthread.h>
#include <iostream>
struct for_index{
    int start;
    int end;
    int increment;
    void *args;
};
void parallel_for(int start, int end, int increment, void *(*functor)(void *), void *arg, int num_threads);
