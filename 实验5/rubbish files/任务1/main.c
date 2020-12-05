# include <stdlib.h>
# include <stdio.h>
# include <math.h>
#include<omp.h>
#include <unistd.h>
#include<pthread.h>
#include"parallel_for.h"
#include"parallel_function.c"


extern double mean;
extern double diff;

int main ( int argc, char *argv[] )
{
	int M=atoi(argv[1]);
	int N=atoi(argv[2]);
   int threadNums=atoi(argv[3]);

  double epsilon = 0.001;
  int iterations;
  int iterations_print;
  double **w;
  double **u;
    w=(double**) malloc (sizeof(double*)*M);
    u=(double**) malloc (sizeof(double*)*M);
  for(int i=0;i<M;i++)
  {
  w[i]=(double*) malloc(sizeof(double)*N);
   u[i]=(double*) malloc(sizeof(double)*N);
  }
  double wtime;

  printf ( "\n" );
  printf ( "HEATED_PLATE_OPENMP\n" );
  printf ( "  C/Pthread version\n" );
  printf ( "  A program to solve for the steady state temperature distribution\n" );
  printf ( "  over a rectangular plate.\n" );
  printf ( "\n" );
  printf ( "  Spatial grid of %d by %d points.\n", M, N );
  printf ( "  The iteration will be repeated until the change is <= %e\n", epsilon );
 printf ( "  Number of processors available = %d\n", omp_get_num_procs ( ) );
  printf ( "  Number of threads =              %d\n", omp_get_max_threads ( ) ); 

/*
  Set the boundary values, which don't change. 
*/
  Arg arg;
 arg.start=1;
 arg.end=N-1;
 arg.pos=0;
 arg.threadNums=threadNums;
 arg.W=w;
arg.target=100;

 paraller_for(0,threadNums,1,changeRow,(void*)&arg,threadNums);
 paraller_for(0,threadNums,1,addRow,(void*)&arg,threadNums);


 arg.start=0;
 arg.end=M-1;
paraller_for(0,threadNums,1,changeCol,(void*)&arg,threadNums);
paraller_for(0,threadNums,1,addCol,(void*)&arg,threadNums);

arg.pos=N-1;
paraller_for(0,threadNums,1,changeCol,(void*)&arg,threadNums);
paraller_for(0,threadNums,1,addCol,(void*)&arg,threadNums);

arg.end=N;
arg.pos=M-1;
arg.target=0;
 paraller_for(0,threadNums,1,changeRow,(void*)&arg,threadNums);
  paraller_for(0,threadNums,1,addRow,(void*)&arg,threadNums);

  mean = mean / ( double ) ( 2 * M + 2 * N - 4 );
  printf ( "\n" );
  printf ( "  MEAN = %f\n", mean );
/* 
  Initialize the interior solution to the mean value.
*/
arg.start=1;
arg.end=M-1;
arg.pos=N-1;
arg.target=mean;
 paraller_for(0,threadNums,1,changeMat,(void*)&arg,threadNums);
/*
  iterate until the  new solution W differs from the old solution U
  by no more than EPSILON.
*/
  iterations = 0;
  iterations_print = 1;
  printf ( "\n" );
  printf ( " Iteration  Change\n" );
  printf ( "\n" );
  wtime = omp_get_wtime ( );

  diff = epsilon;

 while ( epsilon <= diff )
  {
/*
  Save the old solution in U.
*/
arg.start=0;
arg.end=M;
arg.pos=N;
arg.W=w;
arg.U=u;
 paraller_for(0,threadNums,1,copyMat,(void*)&arg,threadNums);
/*
  Determine the new estimate of the solution at the interior points.
  The new solution W is the average of north, south, east and west neighbors.
*/
diff=0;
arg.start=1;
arg.end=M-1;
arg.pos=N-1;
paraller_for(0,threadNums,1,compute,(void*)&arg,threadNums);
paraller_for(0,threadNums,1,findDiff,(void*)&arg,threadNums);
    iterations++;
    if ( iterations == iterations_print )
    {
      printf ( "  %8d  %f\n", iterations, diff );
      iterations_print = 2 * iterations_print;
    }

  } 
  wtime = omp_get_wtime ( ) - wtime + 10;

for(int i=0;i<M;i++)
{
  free (u[i]);
  free (w[i]);
}
free (u);
free (w);
  printf ( "\n" );
  printf ( "  %8d  %f\n", iterations, diff );
  printf ( "\n" );
  printf ( "  Error tolerance achieved.\n" );
  printf ( "  Wallclock time = %f\n", wtime );
/*
  Terminate.
*/
  printf ( "\n" );
  printf ( "HEATED_PLATE_OPENMP:\n" );
  printf ( "  Normal end of execution.\n" );

  return 0;

}
