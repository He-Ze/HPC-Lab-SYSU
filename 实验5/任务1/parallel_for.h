#include <stdlib.h>
#include <stdio.h>
#include <math.h>
#include <omp.h>
#include <pthread.h>
typedef struct{
	int start;
	int end;
	int pos;
	int threadNums;
	int rank;
	double ** W;
	double ** U;
	double target; 
}Arg;
typedef struct{
	int start;
	int end;
	int increment;
	void*(*functor)(void*);
	void *arg;
}parg;
void * toDo(void* arg){
	parg myArg=*(parg*)arg;
	for(int i=myArg.start;i<myArg.end;i+=myArg.increment){
		(*myArg.functor)(myArg.arg);
	}
}
void * paraller_for(int start,int end,int increment,void*(*functor)(void*),void *arg,int num_threads); 