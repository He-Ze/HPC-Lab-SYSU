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

/******************************************************************************/
/*
Purpose:

MAIN is the main program for HEATED_PLATE_OPENMP.

Discussion:   

This code solves the steady state heat equation on a rectangular region.

The sequential version of this program needs approximately
18/epsilon iterations to complete. 


The physical region, and the boundary conditions, are suggested
by this diagram;

W = 0
+------------------+
|                  |
W = 100  |                  | W = 100
|                  |
+------------------+
W = 100

The region is covered with a grid of M by N nodes, and an N by N
array W is used to record the temperature.  The correspondence between
array indices and locations in the region is suggested by giving the
indices of the four corners:

I = 0
[0][0]-------------[0][N-1]
|                  |
J = 0  |                  |  J = N-1
|                  |
[M-1][0]-----------[M-1][N-1]
I = M-1

The steady state solution to the discrete heat equation satisfies the
following condition at an interior grid point:

W[Central] = (1/4) * ( W[North] + W[South] + W[East] + W[West] )

where "Central" is the index of the grid point, "North" is the index
of its immediate neighbor to the "north", and so on.

Given an approximate solution of the steady state heat equation, a
"better" solution is given by replacing each interior point by the
average of its 4 neighbors - in other words, by using the condition
as an ASSIGNMENT statement:

W[Central]  <=  (1/4) * ( W[North] + W[South] + W[East] + W[West] )

If this process is repeated often enough, the difference between successive 
estimates of the solution will go to zero.

This program carries out such an iteration, using a tolerance specified by
the user, and writes the final estimate of the solution to a file that can
be used for graphic processing.

Licensing:

This code is distributed under the GNU LGPL license. 

Modified:

18 October 2011

Author:

Original C version by Michael Quinn.
This C version by John Burkardt.

Reference:

Michael Quinn,
Parallel Programming in C with MPI and OpenMP,
McGraw-Hill, 2004,
ISBN13: 978-0071232654,
LC: QA76.73.C15.Q55.

Local parameters:

Local, double DIFF, the norm of the change in the solution from one iteration
to the next.

Local, double MEAN, the average of the boundary values, used to initialize
the values of the solution in the interior.

Local, double U[M][N], the solution at the previous iteration.

Local, double W[M][N], the solution computed at the latest iteration.
*/

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

  while ( epsilon <= diff ){
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
