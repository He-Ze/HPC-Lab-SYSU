<h1 align=center>中山大学数据科学与计算机学院本科生实验报告</h1>

<h1 align=center>（2020 学年秋季学期）</h1>

   <h2 align=center>课程名称：高性能计算程序设计               任课教师：黄聃</h2>

| 年级+班级 |   18级计科（超算）   | 专业（方向） | 计算机科学与技术（超级计算方向） |
| :-------: | :------------------: | :----------: | :------------------------------: |
|   学号    |       18340052       |     姓名     |               何泽               |
|   Email   | heze_heze@icloud.com |   完成日期   |          2020年11月13日          |

 <h2 align=center>目录</h2>

[TOC]

# Ⅰ  实验目的

## 1. 通过 OpenMP 实现通用矩阵乘法

> 通过 OpenMP 实现通用矩阵乘法(Lab1)的并行版本，OpenMP并行进程从 1 增加至 8，矩阵规模从 512 增加至 2048
>
> 通用矩阵乘法(GEMM)通常定义为:
> $$
> C=AB\\
> C_{m,n}=\sum_{n=1}^NA_{m,n}B_{n,k}
> $$
> 输入：M , N, K 三个整数(512 ~2048)
>
> 问题描述：随机生成 M\*N 和 N*K 的两个矩阵 A,B,对这两个矩阵做乘法得到矩阵 C
>
> 输出：A,B,C 三个矩阵以及矩阵计算的时间

## 2. 基于 OpenMP 的通用矩阵乘法优化

> 分别采用 OpenMP 的默认任务调度机制、静态调度 `schedule(static, 1)` 和动态调度` schedule(dynamic,1)`的性能，实现`#pragma omp for`，并比较其性能。

## 3. 构造基于 Pthreads 的并行 for 循环分解、分配和执行机制。

> ① 基于 pthreads 的多线程库提供的基本函数，如线程创建、线程 join、线程同步等。构建` parallel_for `函数对循环分解、分 配和执行机制，函数参数包括但不限于(`int start, int end, int increment, void *(*functor)(void*), void *arg , int num_threads`);其中 `start` 为循环开始索引;`end `为结束索引; `increment `每次循环增加索引数;`functor `为函数指针，指向的需要被并行执行循环程块;`arg` 为 `functor `的入口参数; `num_threads `为并行线程数。
>
> ② 在 Linux 系统中将 `parallel_for` 函数编译为`.so `文件，由其他程序调用。
>
> ③ 将基于 OpenMP 的通用矩阵乘法的 `omp parallel for `并行，改造成基于 `parallel_for `函数并行化的矩阵乘法，注意只改造可被并行执行的 for 循环(例如无 race condition、无数据依赖、 无循环依赖等)。
>
> 举例说明：
>
> 将串行代码:
>
> ```c
> for ( int i = 0; i < 10; i++ ){
> 	A[i]=B[i] * x + C[i] 
> }
> ```
>
> 替换为：
>
> ```c
> parallel_for(0, 10, 1, functor, NULL, 2); 
> struct for_index {
>        int start;
>        int end;
>        int increment;
> }
> void * functor (void * args){
>     struct for_index * index = (struct for_index *) args;
>     for (int i = index->start; i < index->end; i = i + index->increment){
>         A[i]=B[i] * x + C[i]; 
>     }
> }
> ```
>
> 编译后执行阶段，多线程执行，在两个线程情况下:
>
> Thread0: start 和 end 分别为 0，5 
>
> Thread1: start 和 end 分别为 5，10
>
> ```c
> void * funtor(void * arg){
> 	int start = my_rank * (10/2);
>     int end = start + 10/2; 
>     for(int j = start, j < end, j++) 
>         A[j]=B[j] * x + C[j];
> }
> ```

# Ⅱ   实验过程和核心代码

## 0. 代码整体说明

- 开头`#define isprint 0`这句话定义了一个宏，表示是否打印所有矩阵的元素，为0则不打印，只输出运行时间，为1则打印所有矩阵的所有元素

- 变量`m,n,k`分别代表三个矩阵的行数和列数；三个数组`a,b,c`，`a,b`为待乘矩阵，`c`为结果矩阵

- 我设计的`m,n,k`的值并非在运行时由`scanf`输入，而是在终端运行时由命令行参数输入，赋值的代码如下


## 1. 通过 MPI 实现通用矩阵乘法

### （1）算法简述

$$
C=AB \\
C_{m,n}=\sum_{n=1}^{N}A_{m,n}B_{n,k}
$$

### （2）串行版本

```c++
void gemm(double* A,double* B,double* C,const int M,const int N,const int K)
{
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

### （3） MPI点对点通信实现



## 2. 基于 MPI 的通用矩阵乘法优化

> 点对点通信在前面已经完成，这部分使用集合通信，整体思路和前面一样



# Ⅲ  实验结果

我的电脑CPU有四核八线程，如下图：

![截屏2020-10-03 14.25.46](/Users/heze/Library/Mobile Documents/com~apple~CloudDocs/大三上/高性能计算实验/实验2/图片/截屏2020-10-03 14.25.46.png)

## 1. 通过 MPI 实现通用矩阵乘法

### （1） 验证算法正确性

打印所有矩阵元素，可以通过手算证明结果是正确的



### （2） 测试时间

并行进程从 1 增加至 8，矩阵规模从 512 增加至 2048，所用时间如下图所示：


## 2. 基于 MPI 的通用矩阵乘法优化


## 3. 改造 Lab1 成矩阵乘法库函数



# Ⅳ  实验感想



