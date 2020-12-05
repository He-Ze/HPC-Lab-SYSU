#include"parallel_for.h"

typedef struct
{
	int start;
	int end;
	int increment;
	void*(*functor)(void*);
	void *arg;
}parg;

typedef struct
{
    int start;
    int end;
    int pos;
    int threadNums;
	int rank;
    double ** W;
    double ** U;
    double target; 
}Arg;

void * toDo(void* arg)
{
	parg myArg=*(parg*)arg;
	
	for(int i=myArg.start;i<myArg.end;i+=myArg.increment)
	{
		(*myArg.functor)(myArg.arg);

	}
}

void * paraller_for
(int start,int end,int increment,void*(*functor)(void*),void *arg,int num_threads) 
{
	if((end-start)/(increment*num_threads)<1)
	{
		num_threads=(end-start)/increment;
	}
	
	
	pthread_t  thread_handles[num_threads];
	Arg args[num_threads];
	for(int i=0;i<num_threads;i++)
	{
	args[i]=*(Arg*)arg;
	args[i].rank=i;
	}

	parg tmp[num_threads];
	int local_size=(end-start)/num_threads;
	
	
	for(int i=0;i<num_threads-1;i++)
	{
		 tmp[i];
		tmp[i].start=i*local_size;
		tmp[i].end=tmp[i].start+local_size;
		tmp[i].increment=increment;
		tmp[i].arg=(void*)&args[i];
		tmp[i].functor=functor;
		pthread_create(&thread_handles[i],NULL,toDo,(void*)&tmp[i]);
	}
		 tmp[num_threads-1];
		tmp[num_threads-1].start=(num_threads-1)*local_size;
		tmp[num_threads-1].end=end;
		tmp[num_threads-1].increment=increment;
		tmp[num_threads-1].arg=(void*)&args[num_threads-1];
		tmp[num_threads-1].functor=functor;
		pthread_create(&thread_handles[num_threads-1],NULL,toDo,(void*)&tmp[num_threads-1]);
		
		for(int i=0;i<num_threads;i++)
		{
			pthread_join(thread_handles[i],NULL);
		}
	
}


