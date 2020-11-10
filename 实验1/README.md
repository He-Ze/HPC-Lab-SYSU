<h1 align=center>中山大学数据科学与计算机学院本科生实验报告</h1>

<h1 align=center>（2020 学年秋季学期）</h1>

   <h2 align=center>课程名称：高性能计算程序设计               任课教师：黄聃</h2>

| 年级+班级 |   18级计科（超算）   | 专业（方向） | 计算机科学与技术（超级计算方向） |
| :-------: | :------------------: | :----------: | :------------------------------: |
|   学号    |       18340052       |     姓名     |               何泽               |
|   Email   | heze_heze@icloud.com |   完成日期   |          2020年9月21日           |

[TOC]

---

# 一、 实验目的

---

## 1.  通用矩阵乘法

> 数学上，一个 $m x n$ 的矩阵是一个由m行n列元素排列成的矩形阵列。 矩阵是高等代数中常见的数学工具，也常见于统计分析等应用数学学科中。矩阵运算是数值分析领域中的重要问题。
>
> 通用矩阵乘法(`GEMM`)通常定义为:
> $$
> C=AB \\
> C_{m,n}=\sum_{n=1}^{N}A_{m,n}B_{n,k}
> $$
>
> ---
>
> 请根据定义用 C 语言实现一个矩阵乘法:
>
> 题目:用 C 语言实现通用矩阵乘法
>
> 输入:M , N, K 三个整数(512 ~2048)
>
> 问题描述:随机生成 M*N 和 N*K 的两个矩阵 A,B,对这两个矩阵做乘法得到矩阵 C.
>
> 输出:A,B,C 三个矩阵以及矩阵计算的时间

## 2.  通用矩阵乘法优化

> 对上述的矩阵乘法进行优化，优化方法可以分为以下两类:
>
> 1) 基于算法分析的方法对矩阵乘法进行优化，典型的算法包括 `Strassen` 算法和 `Coppersmith–Winograd` 算法.
>
> 2) 基于软件优化的方法对矩阵乘法进行优化，如循环拆分向量化和内存重排
>
> 实验要求:对优化方法进行详细描述，并提供优化后的源代码，以及与 GEMM 的计算时间对比

## 3.  进阶:大规模矩阵计算优化

> 进阶问题描述:如何让程序支持大规模矩阵乘法? 考虑两个优化方向
>
> 1) 性能，提高大规模稀疏矩阵乘法性能;
>
> 2) 可靠性,在内存有限的情况下,如何保证大规模矩阵乘法计算完成(M, N, K >> 100000),不触发内存溢出异常。
>
> 对优化方法及思想进行详细描述，提供大规模矩阵计算优化代码可加分。

## 4. References

> [1]{GEMM 优化}	https://jackwish.net/2019/gemm-optimization.html
>
> [2] {矩阵说明}	https://zh.wikipedia.org/wiki/%E7%9F%A9%E9%98%B5

# 二、 实验过程和核心代码

---

## 0. 代码框架说明

本实验使用`c++` 完成，最终的代码文件是`avx.cpp`，整体框架如下图所示：

![截屏2020-09-23 14.30.11](/Users/heze/Pictures/截屏/截屏2020-09-23 14.30.11.png)

- `#define isprint 0`这句话定义了一个宏，表示是否打印所有矩阵的元素，为0则不打印，只输出运行时间，为1则打印所有矩阵的所有元素
- 首先定义全局变量`m,n,k`，分别代表三个矩阵的行数和列数
- 再定义三个全局数组，a,b为待乘矩阵，c为结果矩阵
- 我使用了一共四种方法，分别是通用矩阵乘法、`Strassen` 算法、 `Coppersmith–Winograd` 算法和向量化`avx`优化
- 前三种方法分别写到了三个函数中，最后的`avx`写到了`main`函数中
- `main`函数首先要求输入m,n,k的值，按照输入的维度随机生成小于100的浮点数填入到数组中，之后按照不同方法计算

## 1. 通用矩阵乘法

### （1）算法描述

最简单的逻辑，即：
$$
C=AB \\
C_{m,n}=\sum_{n=1}^{N}A_{m,n}B_{n,k}
$$

### （2） 核心代码

```c++
void gemm(double* matA,double* matB,double* matC,const int M,const int N,const int K)
{
	for (int i = 0; i < M;i++){
		for (int j = 0; j < N;j++){
			double sum = 0;
			for (int k = 0; k < K;k++)
				sum += matA[i*M + k] * matB[k*N + j];
			matC[i*K + j] = sum;
		}
	}
}
```

即按照数学表达式，嵌套三层循环，只不过我稍稍优化了两点：

- 将矩阵正常的`a[i][j]`这种索引方式换成了一维的，即`a[i*m+j]`（m为行数）
- 在最内层循环中并没有直接对矩阵元素进行操作，而是先定义一个变量sum，每次嫁到sum中，计算结束后再把sum赋值给矩阵元素，这样在编译器实际优化时可能将sum优化为一个寄存器，效率会有提高

## 2.  Strassen 算法

### （1）算法描述

- `Strassen`算法使用的是分治的思想，当矩阵的阶很大时会采取一个递推式进行计算相关递推式，描述如下：

- $A_{11}，A_{12}，A_{21}，A_{22}$和$B_{11}，B_{12}，B_{21}，B_{22}$分别为两个乘数A和B矩阵的四个子矩阵，$C_{11}，C_{12}，C_{21}，C_{22}$为最终的结果C矩阵的四个子矩阵

- 分别计算如下表达式：

$$
S_1=B_{12}-B_{22}\\
S_2=A_{11}+A_{12}\\
S_3=A_{21}+A_{22}\\
S_4=B_{21}-B_{11}\\
S_5=A_{11}+A_{22}\\
S_6=B_{11}+B_{22}\\
S_7=A_{12}-A_{22}\\
S_8=B_{21}+B_{22}\\
S_9=A_{11}-A_{21}\\
S_{10}=B_{11}+B_{12}\\
$$

$$
P_1=A_{11}\times S_1\\
P_2=S_2\times B_{22}\\
P_3=S_3\times B_{11}\\
P_4=A_{22}\times S_4\\
P_5=S_5\times S_6\\
P_6=S_7\times S_8\\
P_7=S_9\times S_{10}\\
$$

那么最终的矩阵结果为：
$$
C_{11}=P_5+P_4-P_2+P_6\\
C_{12}=P_1+P_2\\
C_{21}=P_3+P_4\\
C_{22}=P_5+P_1-P_3-P_7
$$
那么只需要将这四个矩阵合并就是最终结果。

### （2） 核心代码

>  代码中注释的变量名和上面算法的不一致，在代码中将S、P换成了代码中函数的名字

- 首先因为是分治，那么在将矩阵分到很小的时候如果还是按照这个算法进行就会很慢，不如直接计算，起不到加速的作用，所以在维度小于64的时候直接调用通用矩阵乘法进行计算

  ```c++
  if ((M <= 64) || (M%2 != 0 ||N%2 != 0 ||K%2!=0))
  		return gemm(matA, matB, matC, M, N, K);
  ```

- 然后按照上面的表达式计算

- 首先计算M1

  ```c++
  //M1 = (A11+A22)*(B11+B22)
  std::vector<double> M1((M / 2) * (N / 2));
  {
  	memset(&M1[0], 0, M1.size()*sizeof(double));
  	//M1_0 = (A11+A22)
  	std::vector<double> M1_0((M / 2) * (K / 2));
  	offset = M*M / 2 + K / 2;
  	for (int i = 0; i < M / 2; i++){
  		for (int j = 0; j < K/2; j++){
  			const int baseIdx = i*M + j;
  			M1_0[i*K/2+j] = matA[baseIdx] + matA[baseIdx + offset];
  		}
  	}
  	//M1_1 = (B11+B22)
  	std::vector<double> M1_1((K / 2) * (N / 2));
  	offset = K*M / 2 + N / 2;
  	for (int i = 0; i < K / 2; i++){
  		for (int j = 0; j < N / 2; j++){
  			const int baseIdx = i*M + j;
  			M1_1[i*N/2+j] = matB[baseIdx] + matB[baseIdx + offset];
  		}
  	}
  	strassen(&M1_0[0], &M1_1[0], &M1[0], M / 2, N / 2, K / 2);
  }
  ```

- ==因为计算M2-M7的代码基本上一样，这里为了节省篇幅就不贴代码了，基本上一样==

- 计算完M1-M7之后进行相应计算填入到C矩阵的四个区域

  ```c++
  for (int i = 0; i < M / 2;i++){
  		for (int j = 0; j < N / 2;j++){
  			const int idx = i*N / 2 + j;
  			//C11 = M1+M4-M5+M7
  			matC[i*M + j] = M1[idx] + M4[idx] - M5[idx] + M7[idx];
  			//C12 = M3+M5
  			matC[i*M + j + N/2] = M3[idx] + M5[idx];
  			//C21 = M2+M4
  			matC[(i+M/2)*M + j] = M2[idx] + M4[idx];
  			//C22 = M1-M2+M3+M6
  			matC[(i+M/2)*M + j + N/2] = M1[idx] - M2[idx] + M3[idx] + M6[idx];
  		}
  }
  ```

这样便计算完成

## 3.  Coppersmith–Winograd  算法

### （1）算法描述

计算：
$$
S1 = A21 + A22\\
S2 = S1 - A11\\
S3 = A11 - A21\\
S4 = A12 - S2\\
$$

$$
T1 = B21 - B11\\
T2 = B22 - T1\\
T3 = B22 - B12\\
T4 = T2 - B21
$$

$$
M1 = A11*B11\\
M2 = A12*B21\\
M3 = S4*B22\\
M4 = A22*T4\\
M5 = S1*T1\\
M6 = S2*T2\\
M7 = S3*T3
$$

$$
U1 = M1 + M2\\
U2 = M1 + M6\\
U3 = U2 + M7\\
U4 = U2 + M5\\
U5 = U4 + M3\\
U6 = U3 - M4\\
U7 = U3 + M5
$$

$$
C11 = U1\\
C12 = U5\\
C21 = U6\\
C22 = U7
$$

### （2） 核心代码

- 首先和Strassen一样，维度小于64使用通用乘法

```c++
if ((M <= 64) || (M % 2 != 0 || N % 2 != 0 || K % 2 != 0))
		return gemm(matA, matB, matC, M, N, K);
```

- 计算S矩阵

```c++
	std::vector<double> S1((M / 2) * (K / 2));
	std::vector<double> S2((M / 2) * (K / 2));
	std::vector<double> S3((M / 2) * (K / 2));
	std::vector<double> S4((M / 2) * (K / 2));
	for (int i = 0; i < M / 2;i++){
		for (int j = 0; j < K / 2;j++){
			const double idx = i*K / 2 + j;
			//S1 = A21 + A22
			S1[idx] = matA[(i + M / 2)*M + j] + matA[(i + M / 2)*M + j + K / 2];
			//S2 = S1 - A11
			S2[idx] = S1[idx] - matA[i*M + j];
			//S3 = A11 - A21
			S3[idx] = matA[i*M + j] - matA[(i + M / 2)*M + j];
			//S4 = A12 - S2
			S4[idx] = matA[i*M + j + K / 2] - S2[idx];
		}
	}
```

- 计算T矩阵

```c++
	std::vector<double> T1((K / 2) * (N / 2));
	std::vector<double> T2((K / 2) * (N / 2));
	std::vector<double> T3((K / 2) * (N / 2));
	std::vector<double> T4((K / 2) * (N / 2));
	for (int i = 0; i < K / 2; i++){
		for (int j = 0; j < N / 2; j++){
			const double idx = i*N / 2 + j;
			//T1 = B21 - B11
			T1[idx] = matB[(i + K / 2)*M + j] - matB[i*M + j];
			//T2 = B22 - T1
			T2[idx] = matB[(i + K / 2)*M + j + N / 2] - T1[idx];
			//T3 = B22 - B12
			T3[idx] = matB[(i + K / 2)*M + j + N / 2] - matB[i*M + j + N / 2];
			//T4 = T2 - B21
			T4[idx] = T2[idx] - matB[(i + K / 2)*M + j];
		}
	}
```

- 计算M矩阵

```c++
	//M1 = A11*B11
	std::vector<double> M1((M / 2) * (N / 2));
	{
		memset(&M1[0], 0, M1.size()*sizeof(double));
		winograd(matA, matB, &M1[0], M / 2, N / 2, K / 2);
	}

	//M2 = A12*B21
	std::vector<double> M2((M / 2) * (N / 2));
	{
		memset(&M2[0], 0, M2.size()*sizeof(double));
		winograd(matA + K / 2, matB + K*M/2, &M2[0], M / 2, N / 2, K / 2);
	}

	//M3 = S4*B22
	std::vector<double> M3((M / 2) * (N / 2));
	{
		memset(&M3[0], 0, M3.size()*sizeof(double));
		winograd(&S4[0], matB + K*M/2 + N / 2, &M3[0], M / 2, N / 2, K / 2);
	}

	//M4 = A22*T4
	std::vector<double> M4((M / 2) * (N / 2));
	{
		memset(&M4[0], 0, M4.size()*sizeof(double));
		winograd(matA + M*M / 2 + K / 2, &T4[0], &M4[0], M / 2, N / 2, K / 2);
	}

	//M5 = S1*T1
	std::vector<double> M5((M / 2) * (N / 2));
	{
		memset(&M5[0], 0, M5.size()*sizeof(double));
		winograd(&S1[0], &T1[0], &M5[0], M / 2, N / 2, K / 2);
	}

	//M6 = S2*T2
	std::vector<double> M6((M / 2) * (N / 2));
	{
		memset(&M6[0], 0, M6.size()*sizeof(double));
		winograd(&S2[0], &T2[0], &M6[0], M / 2, N / 2, K / 2);
	}

	//M7 = S3*T3
	std::vector<double> M7((M / 2) * (N / 2));
	{
		memset(&M7[0], 0, M7.size()*sizeof(double));
		winograd(&S3[0], &T3[0], &M7[0], M / 2, N / 2, K / 2);
	}
```

- 计算U矩阵与最终结果

```c++
	for (int i = 0; i < M / 2; i++){
		for (int j = 0; j < N / 2; j++){
			const double idx = i*N / 2 + j;
			//U1 = M1 + M2
			const auto U1 = M1[idx] + M2[idx];
			//U2 = M1 + M6
			const auto U2 = M1[idx] + M6[idx];
			//U3 = U2 + M7
			const auto U3 = U2 + M7[idx];
			//U4 = U2 + M5
			const auto U4 = U2 + M5[idx];
			//U5 = U4 + M3
			const auto U5 = U4 + M3[idx];
			//U6 = U3 - M4
			const auto U6 = U3 - M4[idx];
			//U7 = U3 + M5
			const auto U7 = U3 + M5[idx];

			//C11 = U1
			matC[i*M + j] = U1;
			//C12 = U5
			matC[i*M + j + N / 2] = U5;
			//C21 = U6
			matC[(i + M / 2)*M + j] = U6;
			//C22 = U7
			matC[(i + M / 2)*M + j + N / 2] = U7;
		}
	}
```

## 4. 循环拆分向量化——使用 AVX 指令

> 我电脑的CPU支持如下指令：
>
> <img src="/Users/heze/Library/Application Support/typora-user-images/image-20200923153605146.png" alt="image-20200923153605146" style="zoom:80%;" />
>
> 所以我使用了`AVX1.0`进行编写

### （1）AVX指令简述

#### ①数据类型

| 数据类型  |              描述               |
| :-------: | :-----------------------------: |
| `__m128`  | 包含`4`个`float`类型数字的向量  |
| `__m128d` | 包含`2`个`double`类型数字的向量 |
| `__m128i` |    包含若干个整型数字的向量     |
| `__m256`  | 包含`8`个`float`类型数字的向量  |
| `__m256d` | 包含`4`个`double`类型数字的向量 |
| `__m256i` |    包含若干个整型数字的向量     |

- 每一种类型，从`2`个下划线开头，接一个`m`，然后是`vector`的位长度。
- 如果向量类型是以`d`结束的，那么向量里面是`double`类型的数字。如果没有后缀，就代表向量只包含`float`类型的数字。
- 整形的向量可以包含各种类型的整形数，例如`char,short,unsigned long long`。也就是说，`__m256i`可以包含`32`个`char`，`16`个`short`类型，`8`个`int`类型，`4`个`long`类型。这些整形数可以是有符号类型也可以是无符号类型。

#### ② 函数命名

`_mm<bit_width>_<name>_<data_type>`

- `<bit_width>`表明了向量的位长度，对于`128`位的向量，这个参数为空，对于`256`位的向量，这个参数为`256`。
- `<name>`描述了内联函数的算术操作。
- `<data_type>`标识函数主参数的数据类型。
  - `ps`包含`float`类型的向量
  - `pd `包含`double`类型的向量
  - `epi8/epi16/epi32/epi64` 包含8位/16位/32位/64位的有符号整数
  - `epu8/epu16/epu32/epu64` 包含8位/16位/32位/64位的无符号整数
  - `si128/si256` 未指定的128位或者256位向量
  - `m128/m128i/m128d/m256/m256i/m256d` 当输入向量类型与返回向量的类型不同时，标识输入向量类型

### （2） 核心代码

```c++
__m256d vec_res = _mm256_setzero_pd();
__m256d vec_1 = _mm256_setzero_pd();
__m256d vec_2 = _mm256_setzero_pd();

for (int i = 0; i < m; i++){
	for (int j = 0; j < m; j++){
		vec_1 = _mm256_set1_pd(a[i][j]);
		for (int k = 0; k < m; k += 4){
			vec_2 = _mm256_load_pd(&b[j][k]);
			vec_res = _mm256_load_pd(&c[i][k]);
			vec_res = _mm256_add_pd(vec_res ,_mm256_mul_pd(vec_1, vec_2));
			_mm256_store_pd(&c[i][k], vec_res);
		}
	}
}
```

- 前面首先定义三个`__m256d`类型向量，并使用` _mm256_setzero_pd()`函数置0
- 仿照通用矩阵乘法，三层循环
- 在第二层和第三层之间使用`_mm256_set1_pd()`函数将a矩阵的一部分取到向量`vec_1`中
- 接下来的第三层循环中可以看到每次循环`k+=4`，这是因为一个`__m256d`类型向量是`256`位，一个`double`类型变量占`64`位，故一个向量中包含4个变量，故每次加4
- 在最内层循环中，首先使用`_mm256_set1_pd()`函数将b矩阵的一部分取到向量`vec_2`中，再将c矩阵的一部分取到结果向量`vec_res`中
- 然后计算两个向量的乘积：`_mm256_mul_pd(vec_1, vec_2)`
- 之后将计算得到的乘积与之前取出的之前的结果使用`_mm256_add_pd`相加
- 得到结果后存到c矩阵中`_mm256_store_pd(&c[i][k], vec_res)`
- 循环结束后计算便完成

# 三、 实验结果

---

## 1. 验证算法正确性

> 为了对比各种算法的结果，这里将打印矩阵元素的宏设置为1，并为了比较，将维数分别设置为8和16

维数8:

![截屏2020-09-23 14.16.12](/Users/heze/Pictures/截屏/截屏2020-09-23 14.16.12.png)

维数16:

![image-20200923173106113](/Users/heze/Library/Application Support/typora-user-images/image-20200923173106113.png)

可以看出四种算法计算得到的结果都是一样的，算法正确

## 2. 时间对比与结果分析

- 将打印矩阵元素的宏关闭，只输出时间，使用编译命令：

```shell
g++-10 -std=c++11 -mavx avx.cpp -o avx
```

![截屏2020-09-23 14.26.54](/Users/heze/Pictures/截屏/截屏2020-09-23 14.26.54.png)

可以看出在维度较大时算法优化要比AVX好，而维度较小时AVX更好，当维度是2048时，算法优化加速比接近10

此外，编译器自带的O3优化也对乘法加速比较明显，开启O3优化后的时间如图：

![截屏2020-09-23 14.31.23](/Users/heze/Pictures/截屏/截屏2020-09-23 14.31.23.png)

可见O3优化对于矩阵乘法的优化效果还是比较明显的，将所有时间绘制在一张图里：

![1](/Users/heze/Library/Mobile Documents/com~apple~CloudDocs/大三上/高性能计算实验/实验1/1.jpg)

虚线是没开O3优化，实线是开启的，将下面优化的部分放大：

![2](/Users/heze/Library/Mobile Documents/com~apple~CloudDocs/大三上/高性能计算实验/实验1/2.jpg)

可以看出对于这几种算法，两种算法优化性能比较接近，但是Coppersmith–Winograd  算法更好一些；在开启O3优化时AVX更好，没开启则不如算法优化。

# 四、 实验感想

---

这次实验针对矩阵乘法做了不同的优化，我了解了两种算法以及关于计算时内存访问的知识，学会了使用AVX指令，在实验过程中也遇到了很多编程上的问题和很多bug，比如一开始只要维数大于512就会段错误，后来发现是我一开始每种方法的结果都用了一个数组存，一旦维数大了就有7个`double a[2048][2048]`这样的数组，于是就爆内存了，改进方法就是声明成全局变量；此外使用AVX的时候错了好几次，主要是一开始没弄清每个变量有多少位导致出错。经过这次实验我对于矩阵乘法的优化有了很深入的了解。