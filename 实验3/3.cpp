#include <iostream>
#include <cstdio>
#include <pthread.h>
#include <semaphore.h>
#include <cmath>

using namespace std;

pthread_cond_t cond;
pthread_mutex_t mutex;

//b^2
void *f1(void *a)
{
	double *b=(double *)a;
	double *c;
	c=new double;
	*c=(*b)*(*b);
	pthread_mutex_lock(&mutex);
	pthread_cond_signal(&cond);
	pthread_mutex_unlock(&mutex);
	pthread_exit((void *)c);
	return (void *)c;
}
//4ac
void *f2(void *a)
{
	pair<double,double> *b=(pair<double,double> *)a;
	double *c;
	c=new double;
	*c=4*(b->first)*(b->second);
	pthread_mutex_lock(&mutex);
	pthread_cond_signal(&cond);
	pthread_mutex_unlock(&mutex);
	pthread_exit((void *)c);
	return (void *)c;
}

int main()
{
	double a,b,c;
	cout<<"一元二次方程组形式为ax^2+bx+c=0，请输入其中的a,b和c： ";
	cin>>a>>b>>c;
	
	int count=2;
	pthread_mutex_init(&mutex,0);
	pthread_mutex_lock(&mutex);
	pthread_cond_init(&cond,NULL);

	pthread_t p1,p2;
	pthread_create(&p1,0,f1,(void *)&b);
	pair<double,double> arg(a,c);
	pthread_create(&p2,0,f2,(void *)&arg);

	while(count>0){
		pthread_cond_wait(&cond,&mutex);
		count--;
	}
	pthread_mutex_unlock(&mutex);
	pthread_cond_destroy(&cond);
	pthread_mutex_destroy(&mutex);

	double *m,*n;
	pthread_join(p1,(void **)&m);//m=b^2
	pthread_join(p2,(void **)&n);//n=4ac
	delete m,n;
	double t=sqrt(*m-*n);
	double x1=(-b+t)/(2*a);
	double x2=(-b-t)/(2*a);
	printf("方程%.1fx^2%.1+fx%.1+f=0的解为x1=%.1f,x2=%.1f。\n",a,b,c,x1,x2);
	return 0;
}