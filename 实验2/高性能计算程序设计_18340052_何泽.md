<h1 align=center>中山大学数据科学与计算机学院本科生实验报告</h1>

<h1 align=center>（2020 学年秋季学期）</h1>

   <h2 align=center>课程名称：高性能计算程序设计               任课教师：黄聃</h2>

| 年级+班级 |   18级计科（超算）   | 专业（方向） | 计算机科学与技术（超级计算方向） |
| :-------: | :------------------: | :----------: | :------------------------------: |
|   学号    |       18340052       |     姓名     |               何泽               |
|   Email   | heze_heze@icloud.com |   完成日期   |          2020年10月3日           |

 <h2 align=center>目录</h2>

[TOC]

# Ⅰ  实验目的

## 1. 通过 MPI 实现通用矩阵乘法

> 通过 MPI 实现通用矩阵乘法(Lab1)的并行版本，MPI 并行进程(rank size)从 1 增加至 8，矩阵规模从 512 增加至 2048. 
>
> 通用矩阵乘法(GEMM)通常定义为:
> $$
> C=AB\\
> C_{m,n}=\sum_{n=1}^NA_{m,n}B_{n,k}
> $$
> 输入:M , N, K 三个整数(512 ~2048)
>
> 问题描述:随机生成 M*N 和 N*K 的两个矩阵 A,B,对这两个矩阵做乘法得到矩阵 C.
>
> 输出:A,B,C 三个矩阵以及矩阵计算的时间

## 2. 基于 MPI 的通用矩阵乘法优化

> 分别采用 MPI 点对点通信和 MPI 集合通信实现矩阵乘法中的进程之间通信，并比较两种实现方式的性能。

## 3. 改造 Lab1 成矩阵乘法库函数

> 将 Lab1 的矩阵乘法改造为一个标准的库函数 `matrix_multiply`(函数实现文件和函数头文件)，输入参数为三个完整定义矩阵(A,B,C)， 定义方式没有具体要求，可以是二维矩阵，也可以是 struct 等。在 Linux 系统中将此函数编译为`.so` 文件，由其他程序调用。

# Ⅱ   实验过程和核心代码

## 0. 代码整体说明

- 开头`#define isprint 0`这句话定义了一个宏，表示是否打印所有矩阵的元素，为0则不打印，只输出运行时间，为1则打印所有矩阵的所有元素

- 变量`m,n,k`分别代表三个矩阵的行数和列数；三个数组`a,b,c`，`a,b`为待乘矩阵，`c`为结果矩阵

- 我设计的`m,n,k`的值并非在运行时由`scanf`输入，而是在终端运行时由命令行参数输入，赋值的代码如下：

    ```c++
    m=atoi(argv[1]);
    n=atoi(argv[2]);
    k=atoi(argv[3]);
    ```

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

> 整体思路为主线程按照线程数将矩阵A以行划分为线程数量的块，并发给每个线程，并将矩阵B全部发给每个线程，每个线程计算完一块之后将这一部分结果发给主进线程，主线程计算最后一块再将所有结果汇总

#### 初始化

```c
MPI_Init(NULL,NULL);
MPI_Comm_rank(MPI_COMM_WORLD, &rank);
MPI_Comm_size(MPI_COMM_WORLD, &numprocs);
line = m/numprocs;	//line为每个进程分到的行数
```

#### 主线程

- 首先对两个矩阵随机赋值

    ```c
    srand((unsigned)time(0));
    for(int i=0;i<m; i++){
        for(int j=0;j<n; j++){
            a[i*n+j] = (double)rand()/ (double)(RAND_MAX)*1000;
        }
    }
    for(int i=0;i<n; i++){
        for(int j=0;j<k; j++){
            b[i*k+j] = (double)rand()/ (double)(RAND_MAX)*1000;
        }
    }
    ```

- 开始计时，将矩阵B全部发送，将矩阵A分块发送

    ```c
    //开始计时	
    start = MPI_Wtime();
    //将矩阵B全部发送
    for(int i=1;i<numprocs;i++){
           MPI_Send( b, n*k, MPI_DOUBLE, i, 0, MPI_COMM_WORLD );
    }
    //将矩阵A分块发送
    for(int i=1;i<numprocs;i++){
        MPI_Send( a + (i-1)*line*n, n*line, MPI_DOUBLE, i, 1, MPI_COMM_WORLD);
    }
    ```

- 计算最后一块

    ```c
    for(int i = (numprocs-1)*line;i<m;i++){
        for(int j=0;j<n;j++){
            double temp = 0;
            for(int t=0;t<n;t++)
                temp += a[i*n+t]*b[t*k+j];
            c[i*k+j] = temp;
        }
    }
    ```

- 将各线程结果汇总并结束计时

    ```c
    for(int t=1;t<numprocs;t++){
        MPI_Recv( ans, line*k, MPI_DOUBLE, t, 3, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
        for(int i=0;i<line;i++){
            for(int j=0;j<k;j++){
                c[ ((t-1)*line + i)*k + j] = ans[i*k+j];
            }
        }
    }
    stop = MPI_Wtime();
    ```

#### 其余线程

先接收数组的值，再计算并将结果发送

```c
MPI_Recv(b, n*k, MPI_DOUBLE, 0, 0, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
MPI_Recv(buffer, n*line, MPI_DOUBLE, 0, 1, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
for(int i=0;i<line;i++){
    for(int j=0;j<n;j++){
        double temp=0;
        for(int t=0;t<n;t++)
            temp += buffer[i*n+t]*b[t*k+j];
        ans[i*k+j] = temp;
    }
}
MPI_Send(ans, line*k, MPI_DOUBLE, 0, 3, MPI_COMM_WORLD);
```

## 2. 基于 MPI 的通用矩阵乘法优化

> 点对点通信在前面已经完成，这部分使用集合通信，整体思路和前面一样

### （1）主线程

- 首先对两个矩阵随机赋值

    ```c
    srand((unsigned)time(0));
    for(int i=0;i<m; i++){
        for(int j=0;j<n; j++){
            a[i*n+j] = (double)rand()/ (double)(RAND_MAX)*1000;
        }
    }
    for(int i=0;i<n; i++){
        for(int j=0;j<k; j++){
            b[i*k+j] = (double)rand()/ (double)(RAND_MAX)*1000;
        }
    }
    ```

- 开始计时，将矩阵A分块发送到各进程，将矩阵B广播到所有进程，计算完毕后再将结果汇总

    ```c
    start = MPI_Wtime();
    MPI_Scatter(a, line*n, MPI_DOUBLE, local_a, line*n, MPI_DOUBLE, 0, MPI_COMM_WORLD );
    MPI_Bcast(b, n*k, MPI_DOUBLE, 0, MPI_COMM_WORLD);
    for(int i= 0; i< m;i++){
        for(int j=0;j<n;j++){
            double temp = 0;
            for(int t=0;t<n;t++)
                temp += a[i*n+t] * b[t*k + j];
            ans[i*k + j ] = temp;
        }
    }
    MPI_Gather( ans, line*k, MPI_DOUBLE, c, line*k, MPI_DOUBLE, 0, MPI_COMM_WORLD );
    ```

- 计算最后一块并停止计时

    ```c
    for(int i = (numprocs-1)*line;i<m;i++){
        for(int j=0;j<n;j++){
            double temp = 0;
            for(int t=0;t<n;t++)
                temp += a[i*n+t]*b[t*k+j];
            c[i*k+j] = temp;
        }
    }
    stop = MPI_Wtime();
    ```

### （2） 其余线程

将矩阵A分块发送到各进程，将矩阵B广播到所有进程，计算完毕后再将结果汇总

```c
double * buffer = new double [n*line];
MPI_Scatter(a, line*n, MPI_DOUBLE, buffer, line*n, MPI_DOUBLE, 0, MPI_COMM_WORLD );
MPI_Bcast( b, n * k, MPI_DOUBLE, 0, MPI_COMM_WORLD );
for(int i=0;i<line;i++){
    for(int j=0;j<n;j++){
        double temp=0;
        for(int t=0;t<n;t++)
            temp += buffer[i*n+t]*b[t*k+j];
        ans[i*k+j] = temp;
    }
}
MPI_Gather(ans, line*k, MPI_DOUBLE, c, line*k, MPI_DOUBLE, 0, MPI_COMM_WORLD );
delete [] buffer;
```

# Ⅲ  实验结果

我的电脑CPU有四核八线程，如下图：

![截屏2020-10-03 14.25.46](/Users/heze/Library/Mobile Documents/com~apple~CloudDocs/大三上/高性能计算实验/实验2/图片/截屏2020-10-03 14.25.46.png)

## 1. 通过 MPI 实现通用矩阵乘法

### （1） 验证算法正确性

打印所有矩阵元素，可以通过手算证明结果是正确的

![截屏2020-10-03 20.20.49](/Users/heze/Library/Mobile Documents/com~apple~CloudDocs/大三上/高性能计算实验/实验2/图片/截屏2020-10-03 20.20.49.png)

### （2） 测试时间

并行进程从 1 增加至 8，矩阵规模从 512 增加至 2048，所用时间如下图所示：

（因为我的CPU有4个核心，如果使用大于4个进程就需要在命令行加上`--oversubscribe`）

![截屏2020-10-03 14.18.38](/Users/heze/Library/Mobile Documents/com~apple~CloudDocs/大三上/高性能计算实验/实验2/图片/截屏2020-10-03 14.18.38.png)

可以看到，因为我的CPU有4个核心，所以小于4个时的时间不断减少，加速比接近线程增加的倍数，大于4个后时间接近。

## 2. 基于 MPI 的通用矩阵乘法优化

### （1） 集合通信的正确性验证

打印所有矩阵元素，可以通过手算证明结果是正确的

![截屏2020-10-03 17.33.31](/Users/heze/Library/Mobile Documents/com~apple~CloudDocs/大三上/高性能计算实验/实验2/图片/截屏2020-10-03 17.33.31.png)

### （2） 点对点通信与集合通信时间比较

![截屏2020-10-03 18.37.02](/Users/heze/Library/Mobile Documents/com~apple~CloudDocs/大三上/高性能计算实验/实验2/图片/截屏2020-10-03 18.37.02.png)

可以看出集合通信相比点对点通信时间要增加很多，点对点性能更好一些。

## 3. 改造 Lab1 成矩阵乘法库函数

> 我选择的函数是实验1中使用的`Coppersmith–Winograd `算法优化（因算法在实验一描述过，这里就不描述算法和算法代码了）。

为改造成库函数并测试，共有如下三个源文件：

<img src="/Users/heze/Library/Mobile Documents/com~apple~CloudDocs/大三上/高性能计算实验/实验2/图片/截屏2020-10-03 21.40.30.png" alt="截屏2020-10-03 21.40.30" style="zoom: 33%;" />

### （1）matrix_multiply.h

算法头文件，不包含实现，`gemm`函数为通用矩阵乘法，`matrix_multiply`函数即为之后要调用的函数，这里除了此函数之外还有`gemm`是因为`matrix_multiply`函数在矩阵维数很小时要调用`gemm`函数。

```c++
#include <stdio.h>
#include <stdlib.h>
#include <time.h>
#include <math.h>
#include <vector>
#include <string.h>

void gemm(double* matA,double* matB,double* matC,const int M,const int N,const int K);
void matrix_multiply(double* matA, double* matB, double* matC, const int M, const int N, const int K);
```

### （2）matrix_multiply.cpp

矩阵乘法实现函数，`Coppersmith–Winograd` 算法实现过程实验一已经叙述过，此文件结构如下（已隐藏实现部分）：

![截屏2020-10-03 21.47.49](/Users/heze/Library/Mobile Documents/com~apple~CloudDocs/大三上/高性能计算实验/实验2/图片/截屏2020-10-03 21.47.49.png)

### （3）multiply.cpp

测试`.so`文件的测试函数，输入`m,n,k`的值后对两个矩阵随机赋值，然后调用`.so`库的乘法函数并计时。

开头需要引入头文件： `#include "matrix_multiply.h"`

```c++
#include "matrix_multiply.h"
int m,n,k;
double a[2048][2048],b[2048][2048],c[2048][2048];
int main()
{
	clock_t start,end;
	printf("请依次输入m，n，k的值（范围512～2048）：");
	scanf("%d%d%d",&m,&n,&k);
	srand((unsigned)time(0));
	for (int i = 0; i < m; i++){
		for (int j = 0; j < n; j++){
			a[i][j] = (double)rand() / (double)(RAND_MAX)*100;
			b[i][j] = (double)rand() / (double)(RAND_MAX)*100;
		}
	}
	start=clock();
	matrix_multiply((double*)&a,(double*)&b,(double*)&c,m,n,k);
	end=clock();
	double endtime=(double)(end-start)/CLOCKS_PER_SEC;
	printf("Coppersmith-Winograd优化矩阵乘法已完成，用时：%f ms.\n",endtime*1000);
	return 0;
}
```

### （4） 生成动态链接库

- 首先编译`matrix_multiply.cpp`

    ```shell
    g++-10 -c -fPIC -o matrix_multiply.o matrix_multiply.cpp
    ```

    - `-c`表示只编译而不连接
    - `-o`选项用于说明输出文件名
    - `-fPIC`表示编译为位置独立的代码；不用此选项的话编译后的代码是位置相关的，因此动态载入时是经过代码拷贝的方式来满足不一样进程的需要，而不能达到真正代码段共享的目的
    - 此步骤将生成`matrix_multiply.o`

- 生成`.so`动态链接库

    ```shell
    g++-10 -shared -o libmatrix_multiply.so matrix_multiply.o
    ```

    - `-shared`表示生成一个动态链接库（让链接器生成T类型的导出符号表，有时候也生成弱链接W类型的导出符号），不用该标志外部程序没法链接至一个可执行文件

### （5）载入动态链接库

- 为了使用动态链接库，编译器需要知道`.h`文件位置

    - 对于`#include "..."`，编译器会在当前路径搜索`.h`文件，也可以使用`-I`选项提供额外的搜索路径
    - 对于`#include <...>`，编译器会在默认`include`搜索路径中寻找

- 编译器还需要知道我们用了哪个库文件，在`g++`中：

    - 使用`-l`选项说明库文件的名字（库文件名开头去掉`lib`）
    - 使用`-L`选项说明库文件所在的路径（这里我使用了`-L. `表示当前路径）
    - 如果没有`-L`选项，`g++`将在默认库文件搜索路径中寻找

- 因为我的链接库和头文件都在同一目录下，所以我的编译命令如下：

    ```shell
    g++-10 -o matrix_multiply multiply.cpp -lmatrix_multiply -L.
    ```

之后便可正常运行

### （5） 运行

使用以上命令生成链接库并使用、运行截图如下：

![截屏2020-10-03 21.34.03](/Users/heze/Library/Mobile Documents/com~apple~CloudDocs/大三上/高性能计算实验/实验2/图片/截屏2020-10-03 21.34.03.png)

可见动态链接库生成成功，可以正常调用运行。

# Ⅳ  实验感想

首先，这次实验让我对MPI编程更加熟练，掌握了点对点通信和集合通信的实现和区别，可以更好地使用MPI进行编程。

此外，我了解了什么是动态链接库，并掌握了生成、载入动态链接库的方法，收获颇丰。

