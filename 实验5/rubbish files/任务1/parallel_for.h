# include <stdlib.h>
# include <stdio.h>
# include <math.h>
#include<omp.h>
#include <unistd.h>
#include<pthread.h>
void * toDo(void* arg);
void * paraller_for(int start,int end,int increment,void*(*functor)(void*),void *arg,int num_threads); 

