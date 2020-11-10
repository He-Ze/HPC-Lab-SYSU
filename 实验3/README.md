<h1 align=center>中山大学数据科学与计算机学院本科生实验报告</h1>

<h1 align=center>（2020 学年秋季学期）</h1>

   <h2 align=center>课程名称：高性能计算程序设计               任课教师：黄聃</h2>

| 年级+班级 |   18级计科（超算）   | 专业（方向） | 计算机科学与技术（超级计算方向） |
| :-------: | :------------------: | :----------: | :------------------------------: |
|   学号    |       18340052       |     姓名     |               何泽               |
|   Email   | heze_heze@icloud.com |   完成日期   |          2020年10月22日          |

[TOC]

# Ⅰ  实验目的

## 1. 通过 Pthreads 实现通用矩阵乘法

>
> 通用矩阵乘法(GEMM)通常定义为:
> $$
> C=AB\\
> C_{m,n}=\sum_{n=1}^NA_{m,n}B_{n,k}
> $$
>
> 输入:M , N, K 三个整数(512 ~2048)
>
> 问题描述:随机生成 M*N 和 N*K 的两个矩阵 A,B,对这两个矩阵做乘法得到矩阵 C.
>
> 输出:A,B,C 三个矩阵以及矩阵计算的时间

## 2. 基于 Pthreads 的数组求和

> - 编写使用多个进程/线程对数组 a[1000]求和的简单程序演示 Pthreads 的用法。创建 n 个线程，每个线程通过共享单元
>
>     $global\_index$ 获取 a 数组的下一个未加元素，注意不能在临界段外访问全局下标$ global\_index$
>
> - 重写上面的例子，使得各进程可以一次最多提取 10 个连续的数， 以组为单位进行求和，从而减少对下标的访问

## 3. Pthreads求解二次方程组的根

> 编写一个多线程程序来求解二次方程组𝑎𝑥^2^ + 𝑏𝑥 + 𝑐 = 0的根，使用下面的公式：
> $$
> x=\frac{-b\pm \sqrt{b^2-4ac}}{2a}
> $$
> 中间值被不同的线程计算，使用条件变量来识别何时所有的线程都完成了计算

## 4. 编写一个Pthreads多线程程序来实现基于monte-carlo​方法的y=x^2​阴影面积估算

> $monte-carlo$方法参考课本137页4.2题和本次实验作业的补充材料。 估算$y=x^2$曲线与x轴之间区域的面积，其中x的范围为[0,1]。
>
> <img src="/Users/heze/Library/Application Support/typora-user-images/image-20201020223448545.png" alt="image-20201020223448545" style="zoom: 33%;" />

# Ⅱ  实验过程和核心代码

## 1. 通过 Pthreads 实现通用矩阵乘法

### （1） 前置代码说明

- 开头`#define isprint 0`这句话定义了一个宏，表示是否打印所有矩阵的元素，为0则不打印，只输出运行时间，为1则打印所有矩阵的所有元素

- 变量`m,n,k`分别代表三个矩阵的行数和列数；三个数组`a,b,c`，`a,b`为待乘矩阵，`c`为结果矩阵

- 我设计的`m,n,k`的值并非在运行时由`scanf`输入，而是在终端运行时由命令行参数输入，赋值的代码如下：

    ```c++
    m=atoi(argv[1]);
    n=atoi(argv[2]);
    k=atoi(argv[3]);
    ```

- `Pthread` 每个线程运行的函数为`void *thread(void *p){};`

### （2）算法简述

$$
C=AB \\
C_{m,n}=\sum_{n=1}^{N}A_{m,n}B_{n,k}
$$

### （3）串行版本

```c++
void gemm(double* A,double* B,double* C,const int M,const int N,const int K){
	for (int i = 0; i < M;i++){
		for (int j = 0; j < N;j++){
			double sum = 0;
			for (int k = 0; k < K;k++)
				sum += A[i*N + k] * B[k*K + j];
			C[i*K + j] = sum;
		}
	}
}
```

### （4）Pthread版本

> 我的算法整体设计思路是Pthread的每个线程计算结果矩阵的一个元素，也就是每个线程计算矩阵A的某一行与矩阵B的某一列的乘积从而得到结果矩阵的一个元素。

#### 全局变量

```c
double a[2048][2048],b[2048][2048],c[2048][2048];
int m,n,k;
struct v {
	int roll,col;
};
```

- 三个数组`a,b,c`，`a,b`为待乘矩阵，`c`为结果矩阵
- `m,n,k`分别代表三个矩阵的行数和列数
- 由于每个线程计算A的某一行与B的某一列的乘积，那么需要给每个线程传递的参数就是A的行数和B的列数，于是我将这两个变量写到了一个结构体之中，分别是`roll`和`col`

#### 主函数

- 首先对两个矩阵随机赋值

    ```c
    srand((unsigned)time(0));
    for (int i = 0; i < m; i++){
        for (int j = 0; j < n; j++){
            a[i][j] = (double)rand() / (double)(RAND_MAX)*1000;
            b[i][j] = (double)rand() / (double)(RAND_MAX)*1000;
        }
    }
    ```

- 然后对矩阵每一行每一列进行循环，先将行数和列数赋值给结构体，然后对线程属性初始化：`pthread_attr_init(&attr); `，之后创建线程并调用`thread`函数：`pthread_create(&t,&attr,thread,data);` ，之后`pthread_join(t, NULL); `即可

    ```c
    for(int i = 0; i < m; i++) {
        for(int j = 0; j < k; j++) {
            struct v *data = (struct v *) malloc(sizeof(struct v));
            data->roll = i;
            data->col = j;
            pthread_t t;
            pthread_attr_t attr;
            pthread_attr_init(&attr);
            pthread_create(&t,&attr,thread,data);
            pthread_join(t, NULL);
        }
    }
    ```

#### 线程函数

收到行数列数的参数之后，就计算A矩阵这一行和B矩阵这一列的乘积，之后写入C结果矩阵中。

```c
void *thread(void *x) {
	struct v *data = (v*)x; 
	double sum = 0;
	for(int t = 0; t< n; t++){
		sum += a[data->roll][t] * b[t][data->col];
	}
	c[data->roll][data->col] = sum;
	pthread_exit(0);
}
```

## 2. 基于 Pthreads 的数组求和

### A. 每次获取一个元素

#### （1）算法思路

定义全局变量`global_index​` ，每个线程获取索引之后加到全局变量​`global_sum​` 中，并对索引加一，以上操作都在锁的保护下完成，直到对数组所有元素计算完成之后停止。

#### （2）全局变量

```c
int ARRAY_SIZE;
int NUM_THREADS;
int array[10000];
int global_index = 0;
int global_sum = 0;
pthread_mutex_t mutex=PTHREAD_MUTEX_INITIALIZER;
```

- `ARRAY_SIZE`和`NUM_THREADS`分别表示数组元素个数和线程个数
- `array`数组存放待加和元素
- `global_index`和`global_sum`表示全局索引和全局总和
- `pthread_mutex_t mutex=PTHREAD_MUTEX_INITIALIZER;` 对锁进行初始化

#### （3）主函数

- 数组元素个数和线程个数并非直接赋值，而是由运行命令中的参数获取

    ```c
    ARRAY_SIZE=atoi(argv[1]);
    NUM_THREADS=atoi(argv[2]);
    ```

- 声明线程数组，初始化锁并对待加数组随机赋值

    ```c
    pthread_t thread[NUM_THREADS];
    pthread_mutex_init (&mutex, NULL);
    srand((unsigned)time(0));
    for (int i=0;i<ARRAY_SIZE;i++)
        array[i]=(int)rand()%100;
    ```

- 创建线程并调用`add` 线程函数进行加和，由于每个线程只加一个元素，所以每个线程都要循环`元素个数/线程数`才可以计算完所有的元素

    ```c
    for (int i=0;i<NUM_THREADS;i++){
        for (int t=0;t<ARRAY_SIZE/NUM_THREADS;t++) 
            pthread_create(&thread[i], NULL, add, NULL);
    }
    for (int i=0;i<NUM_THREADS;i++){
        for (int t=0;t<ARRAY_SIZE/NUM_THREADS;t++) 
            pthread_join(thread[i], NULL);
    }
    ```

#### （4）线程函数

在锁内获取索引元素加和并对索引加一。

```c
void* add ()
{
	pthread_mutex_lock(&mutex);
	global_sum += array[global_index];
	global_index++;
	pthread_mutex_unlock(&mutex);
}
```

### B. 每次获取十个元素

> 大体思路与前面一致，只不过每个线程函数多了一个局部变量存放10个数字的和，循环次数也有所改变

#### （1）创建线程

内层循环次数由之前的`ARRAY_SIZE/NUM_THREADS`变成了`ARRAY_SIZE/NUM_THREADS/10`

```c
for (int i=0;i<NUM_THREADS;i++){
    for (int t=0;t<ARRAY_SIZE/NUM_THREADS/10;t++) 
        pthread_create(&thread[i], NULL, add, NULL);
}
for (int i=0;i<NUM_THREADS;i++){
    for (int t=0;t<ARRAY_SIZE/NUM_THREADS/10;t++) 
        pthread_join(thread[i], NULL);
}
```

#### （2）线程函数

定义了局部变量`local_sum`每次存放10个数字的和，之后加到全局的和中，之后全局索引加10，均在锁内完成。

```c
void* add (void* index)
{
	int local_sum =0;
	pthread_mutex_lock(&mutex);
	for(int i=0; i< 10; i++)
		local_sum +=array[global_index+i];
	global_sum += local_sum;
	global_index+=10;
	pthread_mutex_unlock(&mutex);
}
```

## 3. Pthreads求解二次方程组的根

### （1）全局变量

将锁和条件变量定义为全局变量

```c
pthread_cond_t cond;
pthread_mutex_t mutex;
```

### （2）线程函数

- 计算b^2^

    ```c
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
    ```

    b^2^计算完成后对条件变量`pthread_cond_signal(&cond);`加锁，之后`pthread_exit((void *)c);并将结果返回`

- 计算4ac

    ```c
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
    ```

    由于要传过来两个变量，于是将a和c写成了一个`pair` ，其余结构与前面一致

### （3）主函数

- 初始化

    读进a,b,c的值之后对锁和条件变量初始化

    ```c
    int count=2;
    pthread_mutex_init(&mutex,0);
    pthread_mutex_lock(&mutex);
    pthread_cond_init(&cond,NULL);
    ```

- 创建线程

    ```c
    pthread_t p1,p2;
    pthread_create(&p1,0,f1,(void *)&b);
    pair<double,double> arg(a,c);
    pthread_create(&p2,0,f2,(void *)&arg);
    ```

- 主线程条件变量的操作

    ```c
    while(count>0){
        pthread_cond_wait(&cond,&mutex);
        count--;
    }
    ```

- 结束线程，并将两个结果赋值给m和n

    ```c
    pthread_mutex_unlock(&mutex);
    pthread_cond_destroy(&cond);
    pthread_mutex_destroy(&mutex);
    double *m,*n;
    pthread_join(p1,(void **)&m);//m=b^2
    pthread_join(p2,(void **)&n);//n=4ac
    ```

- 主线程根据公式完成剩余的步骤，计算两个根并输出

    ```c
    double t=sqrt(*m-*n);
    double x1=(-b+t)/(2*a);
    double x2=(-b-t)/(2*a);
    printf("方程%.1fx^2%.1+fx%.1+f=0的解为x1=%.1f,x2=%.1f。\n",a,b,c,x1,x2);
    ```

## 4. 编写一个Pthreads多线程程序来实现基于monte-carlo​方法的y=x^2​阴影面积估算

### （1）算法描述

按照我的理解，使用`monte-carlo`方法求解y=x^2^与x轴相交部分面积，通俗地讲就是在`1X1` 的正方形内随机生成点，点在y=x^2^下面的个数占所有点的比例就是面积，那么只需要统计随机生成的点有多大几率落到曲线下方即可。

### （2）全局变量

```c
long global_num = 0;
long thread_num;
pthread_mutex_t mutex = PTHREAD_MUTEX_INITIALIZER;
```

- `global_num` 指的是所有线程一共有多少个点落在曲线下方
- `thread_num`指每个线程需要统计多少个点
- `pthread_mutex_t mutex = PTHREAD_MUTEX_INITIALIZER;` 定义锁并初始化

### （3） 线程函数

定义`local_num`表示一个线程中有多少曲线下方的点，之后随机两个0至1的数并统计，在锁内加至全局总和

```c
void *thread() 
{
	long local_num = 0;
	unsigned int a = rand();
	for (long i = 0; i < thread_num; i++) {
		double x = rand_r(&a) / ((double)RAND_MAX + 1)*2.0-1.0;
		double y = rand_r(&a) / ((double)RAND_MAX + 1)*2.0-1.0;
		if (y>x*x)
			local_num++;
	}
	pthread_mutex_lock(&mutex);
	global_num += local_num;
	pthread_mutex_unlock(&mutex);
}
```

### （4）主函数

- 初始化，`totalpoints`表示一共随机多少个点，`thread_count`表示使用多少线程，之后定义锁属性并初始化

    ```c
    long totalpoints = 1000000000;
    int thread_count = 10000;
    thread_num = totalpoints/thread_count;
    srand((unsigned)time(NULL));
    pthread_t *threads = malloc(thread_count * sizeof(pthread_t));
    pthread_attr_t attr;
    pthread_attr_init(&attr);
    ```

- 线程操作，调用前面的`thread`函数，最后计算并打印最终结果

    ```c
    for (int i = 0; i < thread_count; i++) {
        pthread_create(&threads[i], &attr, thread, (void *) NULL);
    }
    for (int i = 0; i < thread_count; i++) {
        pthread_join(threads[i], NULL);
    }
    pthread_mutex_destroy(&mutex);
    free(threads);
    printf("x在0～1时y=x^2与x轴之间区域的面积是%f\n", (double)global_num/(double)totalpoints);
    ```

# Ⅲ  实验结果

## 1. 通过 Pthreads 实现通用矩阵乘法

### （1）验证结果正确性

打印所有元素，可以通过手算验证结果是正确的：

<img src="/Users/heze/Pictures/截屏/截屏2020-10-21 18.45.49.png" alt="截屏2020-10-21 18.45.49" style="zoom: 50%;" />

## （2）结果与时间比较

<img src="/Users/heze/Pictures/截屏/截屏2020-10-23 16.31.20.png" alt="截屏2020-10-23 16.31.20" style="zoom:67%;" />

上面是Pthread的结果，实验1用其他方法的结果如下图：

<img src="/Users/heze/Pictures/截屏/截屏2020-10-23 16.34.10.png" alt="截屏2020-10-23 16.34.10" style="zoom: 45%;" />

可以看出由于我没有做任何优化，每个线程只计算一个元素导致线程开销很大，运行很慢。

## 2. 基于 Pthreads 的数组求和

### （1）验证算法正确性

只计算很少的元素，并全部打印出来，可以通过手算验证算法是正确的

<img src="/Users/heze/Pictures/截屏/截屏2020-10-22 16.57.11.png" alt="截屏2020-10-22 16.57.11" style="zoom:67%;" />

计算1000个元素：

<img src="/Users/heze/Pictures/截屏/截屏2020-10-22 16.58.20.png" alt="截屏2020-10-22 16.58.20" style="zoom:67%;" />

### （2）时间比较

每次只取一个元素与单纯`for` 循环进行比较：

<img src="/Users/heze/Pictures/截屏/截屏2020-10-22 17.29.57.png" alt="截屏2020-10-22 17.29.57" style="zoom:67%;" />

每次取十个元素与单纯`for` 循环进行比较：

<img src="/Users/heze/Pictures/截屏/截屏2020-10-22 21.18.27.png" alt="截屏2020-10-22 21.18.27" style="zoom:67%;" />

**由上面两个图比较可以看出，每个线程每次取十个元素要比取一个元素的效率有很大的提升，但是由于线程的开销和编译器对for循环的优化，还是要比直接for循环慢很多**

## 3. Pthreads求解二次方程组的根

- $x^2-2x+1=0$

<img src="/Users/heze/Pictures/截屏/截屏2020-10-22 23.33.59.png" alt="截屏2020-10-22 23.33.59" style="zoom:67%;" />

- $x^2-4x+4=0$

<img src="/Users/heze/Pictures/截屏/截屏2020-10-22 23.34.16.png" alt="截屏2020-10-22 23.34.16" style="zoom:67%;" />

- $4x^2-4x+1=0$

<img src="/Users/heze/Pictures/截屏/截屏2020-10-22 23.35.23.png" alt="截屏2020-10-22 23.35.23" style="zoom:67%;" />

**由此可以看出算法正确，解出的根全部正确。**

## 4. 编写一个Pthreads多线程程序来实现基于monte-carlo​方法的y=x^2​阴影面积估算

<img src="/Users/heze/Pictures/截屏/截屏2020-10-23 09.03.18.png" alt="截屏2020-10-23 09.03.18" style="zoom:67%;" />

由数学知识，$y=x^2$ 在$x\in[0,1]$ 与x轴之间区域面积为$\int_{0}^1x^2=\frac13x^3|_0^1=\frac13$ ，约为0.33，上面计算得到的结果与0.33接近，结果正确。


# Ⅳ  实验感想

这次实验围绕Pthread设置了很多题目，全都完成之后我对Pthread的编程方法、锁和条件变量都有了很深入的认识和理解，了解了如何使用他们并处理互斥问题，加深了理解。