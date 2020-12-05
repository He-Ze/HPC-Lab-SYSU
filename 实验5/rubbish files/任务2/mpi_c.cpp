# include <stdlib.h>
# include <stdio.h>
# include <math.h>
# include <omp.h>
# include <mpi.h>

int main ( int argc, char *argv[] );

/******************************************************************************/



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
 
 double **w;
  double **u;
    w=(double**) malloc (sizeof(double*)*M);
    u=(double**) malloc (sizeof(double*)*M);
  for(int i=0;i<M;i++)
  {
  w[i]=(double*) malloc(sizeof(double)*N);
   u[i]=(double*) malloc(sizeof(double)*N);
  }


  double diff;
  double epsilon = 0.001;
  int i;
  int iterations;
  int iterations_print;
  int j;
  double mean;
  double my_diff;

  double wtime;

  printf ( "\n" );
  printf ( "HEATED_PLATE_OPENMP\n" );
  printf ( "  C/OpenMP version\n" );
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
  mean = 0.0;

	int myrank, numprocs;

    MPI_Status status;
	
    MPI_Init(&argc, &argv);  // 并行开始
    MPI_Comm_size(MPI_COMM_WORLD, &numprocs); 
    MPI_Comm_rank(MPI_COMM_WORLD, &myrank); 


#pragma omp parallel shared ( w ) private ( i, j )
  {
#pragma omp for
    for ( i = 1; i < M - 1; i++ )
    {
      w[i][0] = 100.0;
    }
#pragma omp for
    for ( i = 1; i < M - 1; i++ )
    {
      w[i][N-1] = 100.0;
    }
#pragma omp for
    for ( j = 0; j < N; j++ )
    {
      w[M-1][j] = 100.0;
    }
#pragma omp for
    for ( j = 0; j < N; j++ )
    {
      w[0][j] = 0.0;
    }
/*
  Average the boundary values, to come up with a reasonable
  initial value for the interior.
*/
#pragma omp for reduction ( + : mean )
    for ( i = 1; i < M - 1; i++ )
    {
      mean = mean + w[i][0] + w[i][N-1];
    }
#pragma omp for reduction ( + : mean )
    for ( j = 0; j < N; j++ )
    {
      mean = mean + w[M-1][j] + w[0][j];
    }
  }
/*
  OpenMP note:
  You cannot normalize MEAN inside the parallel region.  It
  only gets its correct value once you leave the parallel region.
  So we interrupt the parallel region, set MEAN, and go back in.
*/
  mean = mean / ( double ) ( 2 * M + 2 * N - 4 );
  printf ( "\n" );
  printf ( "  MEAN = %f\n", mean );
/* 
  Initialize the interior solution to the mean value.
*/
#pragma omp parallel shared ( mean, w ) private ( i, j )
  {
#pragma omp for
    for ( i = 1; i < M - 1; i++ )
    {
      for ( j = 1; j < N - 1; j++ )
      {
        w[i][j] = mean;
      }
    }
  }
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
	if(my_rank==0) 
	{ 
		for (int i = 0; i < M; i++ ) 
		{ 
			for (int j = 0; j < N; j++ ) 
			{ 
				u[i][j] = w[i][j]; 
			} 
		}
		for(int i=1;i<=comm_sz-1;i++) 
		{ 
			int start=1+(i-1)*size; 
			int pos=0; 
			int end; 
			if(i!=comm_sz-1) 
			{ 
				end=start+size; 
			}
			else 
			{ 
				end=M-1; 
			}
			for(int j=start-1;j<end+1;j++) 
			{ 
				MPI_Pack(u[j],N,MPI_DOUBLE,buf,sizeof(double)* ((size+4)*N),&pos,MPI_COMM_WORLD); 
			}
			MPI_Send(buf,pos,MPI_PACKED,i,0,MPI_COMM_WORLD); 
		}
		for(int i=1;i<comm_sz;i++) 
		{ 
			int pos=0; 
			MPI_Recv(buf,sizeof(double)* ((size+4)*N),MPI_PACKED,i,0,MPI_COMM_WORLD,MPI_STATUS_IGNORE); 
		MPI_Unpack(buf,sizeof(double)* ((size+4)*N),&pos,&di[i],1,MPI_DOUBLE,MPI_COMM_WORLD); 
			int start=1+(i-1)*size; 
			int rows; 
			if(i!=comm_sz-1) 
			{ 
				rows=size; 
			}
			else 
			{ 
				rows=M-1-start; 
			}
			int j=start; 
			for(int i=0;i<rows;i++) 
			{ 
				MPI_Unpack(buf,sizeof(double)* ((size+4)*N),&pos,w[j],N,MPI_DOUBLE,MPI_COMM_WORLD); 
				j++; 
			} 
		}
	}
	else
	{
		MPI_Recv(buf,sizeof(double)* ((size+4)*N),MPI_PACKED,0,0,MPI_COMM_WORLD,MPI_STATUS_IGNORE); 		int pos=0; 
		int start=1+(my_rank-1)*size; 
		int rows; 
		if(my_rank!=comm_sz-1) 
		{ 
			rows=size+2; 
		}
		else 
		{ 
			rows=M-start+1; 
		} 
		for(int i=0;i<rows;i++) 
		{
			MPI_Unpack(buf,sizeof(double)* ((size+4)*N),&pos,tmpU[i],N,MPI_DOUBLE,MPI_COMM_WORLD); 
		}
		for(int i=1;i<rows-1;i++) 
		{
			for(int j=1;j<N-1;j++) 
			{ 
				tmpW[i][j]=( tmpU[i-1][j] + tmpU[i+1][j] + tmpU[i][j-1] + tmpU[i][j+1] ) / 4.0; 
			}
			tmpW[i][0]=tmpW[i][N-1]=100; 
		}
		diff=0; 
		for (int i = 1; i < rows - 1; i++ ) 
		{ 
			for (int j = 1; j < N - 1; j++ ) 
			{ 
				if ( diff < fabs ( tmpW[i][j] - tmpU[i][j] ) ) 
				{ 
					diff = fabs ( tmpW[i][j] - tmpU[i][j] ); 
				} 
			} 
		}
		pos=0; 
		MPI_Pack(&diff,1,MPI_DOUBLE,buf,sizeof(double)* ((size+4)*N),&pos,MPI_COMM_WORLD); 
		for (int i = 1; i < rows - 1; i++ ) 
		{ 
			MPI_Pack(tmpW[i],N,MPI_DOUBLE,buf,sizeof(double)* ((size+4)*N),&pos,MPI_COMM_WORLD); 
		}
		MPI_Send(buf,pos,MPI_PACKED,0,0,MPI_COMM_WORLD); 
	}
  }
  wtime = omp_get_wtime ( ) - wtime;
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
