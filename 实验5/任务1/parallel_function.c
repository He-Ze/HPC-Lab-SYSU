#include"parallel_for.h"
pthread_mutex_t  myLock = PTHREAD_MUTEX_INITIALIZER;
double mean=0,diff=0;

void *changeRow(void * a){
    Arg arg=*(Arg*)a;
    int size=(arg.end-arg.start)/arg.threadNums;
    int rank=arg.rank;
    int start=arg.start+size*rank;
    int end;
    if( rank!=arg.threadNums-1)
        end=start+size;
    else
        end=arg.end;
    for(int i=start;i<end;i++)
        arg.W[arg.pos][i]=arg.target;
}

void *addRow(void * a){
    Arg arg=*(Arg*)a;
    int size=(arg.end-arg.start)/arg.threadNums;
    int rank=arg.rank;
    int start=arg.start+size*rank;
    int end;
    if( rank!=arg.threadNums-1)
      end=start+size;
    else
      end=arg.end;
    double tmp;
    for(int i=start;i<end;i++)
      tmp+= arg.W[arg.pos][i];
    pthread_mutex_lock(&myLock);
    mean+=tmp;
    pthread_mutex_unlock(&myLock);
}

void *changeCol(void *a){
    Arg arg=*(Arg*)a;
    int size=(arg.end-arg.start)/arg.threadNums;
    int rank=arg.rank;
    int start=arg.start+size*rank;
    int end;
    if( rank!=arg.threadNums-1)
        end=start+size;
    else
        end=arg.end;
    for(int i=start;i<end;i++)
        arg.W[i][arg.pos]=arg.target;
}

void *addCol(void *a){
    Arg arg=*(Arg*)a;
    int size=(arg.end-arg.start)/arg.threadNums;
    int rank=arg.rank;
    int start=arg.start+size*rank;
    int end;
    if( rank!=arg.threadNums-1)
      end=start+size;
    else
      end=arg.end;
    double tmp;
    for(int i=start;i<end;i++)
      tmp+= arg.W[i][arg.pos];
    pthread_mutex_lock(&myLock);
    mean+=tmp;
    pthread_mutex_unlock(&myLock);
}

void *changeMat(void *a){
    Arg arg=*(Arg*)a;
    int size=(arg.end-arg.start)/arg.threadNums;
    int rank=arg.rank;
    int start=arg.start+size*rank;
    int end;
    if( rank!=arg.threadNums-1)
      end=start+size;
    else
      end=arg.end;
    for(int i=start;i<end;i++)
      for(int j=1;j<arg.pos;j++)
      arg.W[i][j]=arg.target;
}

void *copyMat(void *a){
    Arg arg=*(Arg*)a;
    int size=(arg.end-arg.start)/arg.threadNums;
    int rank=arg.rank;
    int start=arg.start+size*rank;
    int end;
    if( rank!=arg.threadNums-1)
        end=start+size;
    else
        end=arg.end;
    for(int i=start;i<end;i++)
        for(int j=0;j<arg.pos;j++)
            arg.U[i][j]=arg.W[i][j];
}

void * compute(void *a){
    Arg arg=*(Arg*)a;
    int size=(arg.end-arg.start)/arg.threadNums;
    int rank=arg.rank;
    int start=arg.start+size*rank;
    int end;
    double **w=arg.W;
    double **u=arg.U;
    if( rank!=arg.threadNums-1)
        end=start+size;
    else
        end=arg.end;
    for(int i=start;i<end;i++)
        for(int j=1;j<arg.pos;j++)
            w[i][j] = ( u[i-1][j] + u[i+1][j] + u[i][j-1] + u[i][j+1] ) / 4.0;
}

void * findDiff(void *a){
    int key1,key2;
    Arg arg=*(Arg*)a;
    int size=(arg.end-arg.start)/arg.threadNums;
    int rank=arg.rank;
    int start=arg.start+size*rank;
    int end;
    double **w=arg.W;
    double **u=arg.U;
    if( rank!=arg.threadNums-1)
        end=start+size;
    else
        end=arg.end;
    double myDiff=0.0;
    for(int i=start;i<end;i++){
        for(int j=1;j<arg.pos;j++){
              if(myDiff<fabs(w[i][j]-u[i][j])){
                  myDiff=fabs(w[i][j]-u[i][j]);
                  key1=i;
                  key2=j;
              }
        }
    }
    pthread_mutex_lock(&myLock);
    if ( diff < myDiff )
        diff = myDiff;
    pthread_mutex_unlock(&myLock);
}


