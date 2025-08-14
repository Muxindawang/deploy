## 基本概念

- 延迟：指操作从开始到结束所需要的时间，一般用微秒计算，延迟越低越好。
- 带宽：单位时间内处理的数据量，一般用MB/s或者GB/s表示。
- 吞吐量：位时间内成功处理的运算数量，一般用gflops来表示（十亿次浮点计算），吞吐量和延迟有一定关系，都是反应计算速度的，一个是时间除以运算次数，得到的是单位次数用的时间–延迟，一个是运算次数除以时间，得到的是单位时间执行次数–吞吐量。

CPU适合执行复杂的逻辑，比如多分支，其核心比较重（复杂）；GPU适合执行简单的逻辑，大量的数据计算，其吞吐量更高，但是核心比较轻（结构简单）；低并行逻辑复杂的程序适合用CPU；高并行逻辑简单的大数据计算适合GPU。

![img](https://face2ai.com/CUDA-F-1-1-%E5%BC%82%E6%9E%84%E8%AE%A1%E7%AE%97-CUDA/2.png)



一般CUDA程序分成下面这些步骤：

1. 分配GPU内存
2. 拷贝内存到设备
3. 调用CUDA内核函数来执行计算
4. 把计算完成数据拷贝回主机端
5. 内存销毁

Thread、Block、Grid 的概念：

- `Thread`: 一个 CUDA Kernel 可以被多个 threads 来执行
- `Block`: 多个 threads 会组成一个 Block，而同一个 block 中的 threads 可以同步，也可以通过 shared memory 通信
- `Grid`: 多个 blocks 可以组成一个 Grid

其中，一个 Grid 可以包含多个 Blocks。Blocks 的分布方式可以是一维的，二维，三维的；Block 包含多个 Threads，Threads 的分布方式也可以是一维，二维，三维的。

**所有CUDA核函数的启动都是异步的；一个核函数只能有一个grid，一个grid可以有很多个块，每个块可以有很多的线程；不同块内线程不能相互影响！他们是物理隔离的！**

![img](https://face2ai.com/CUDA-F-2-0-CUDA%E7%BC%96%E7%A8%8B%E6%A8%A1%E5%9E%8B%E6%A6%82%E8%BF%B01/3.png)

![img](https://face2ai.com/CUDA-F-2-0-CUDA%E7%BC%96%E7%A8%8B%E6%A8%A1%E5%9E%8B%E6%A6%82%E8%BF%B01/4.png)

Kernel核函数编写有以下限制

1. 只能访问设备内存
2. 必须有void返回类型
3. 不支持可变数量的参数
4. 不支持静态变量
5. 显示异步行为

### 线程索引

Grid 为 一维，Block 为一维：

```cpp
int threadId = blockIdx.x *blockDim.x + threadIdx.x; 
```

Grid 为 一维，Block 为二维：

```cpp
int threadId = blockIdx.x * blockDim.x * blockDim.y + 
              threadIdx.y * blockDim.x + threadIdx.x;  
```

Grid 为 一维，Block 为三维：

```cpp
Block三维可以这么理解，块->面->行  块的大小是blockDim.x * blockDim.y * blockDim.z，在第blockIdx.x个块中，面的大小是blockDim.y * blockDim.x，第threadIdx.z面，行的大小是blockDim.x，第threadIdx.y行，第threadIdx.x个

int threadId = blockIdx.x * blockDim.x * blockDim.y * blockDim.z + 
              threadIdx.z * blockDim.y * blockDim.x +
              threadIdx.y * blockDim.x + 
    		  threadIdx.x;  
```

Grid 为 二维，Block 为三维：

```cpp
int blockId = blockIdx.x + blockIdx.y * gridDim.x;  
int threadId = blockId * (blockDim.x * blockDim.y * blockDim.z)  
                       + (threadIdx.z * (blockDim.x * blockDim.y))  
                       + (threadIdx.y * blockDim.x) + threadIdx.x;  
```

Grid 为 三维，Block 为三维：

```cpp
先找到属于哪一个Block，再在该Block查找是哪个thread

int blockId = blockIdx.x + blockIdx.y * gridDim.x  
             + gridDim.x * gridDim.y * blockIdx.z;  

int threadId = blockId * (blockDim.x * blockDim.y * blockDim.z)  
                       + (threadIdx.z * (blockDim.x * blockDim.y))  
                       + (threadIdx.y * blockDim.x) + threadIdx.x; 
```



### CUDA层次结构

Block中最多可以有**1024**个线程

块中的线程数可以使用一个通常称为 `blockDim` 的变量进行配置，它是一个由三个整数组成的向量。该向量的条目指定了 `blockDim.x`、`blockDim.y` 和 `blockDim.z` 的大小，如下图所示：

![picture 0](https://cuda.keter.top/assets/images/0b35adb64a964e56018dc9fb7277269a3efa72b1526058609e0860f33e00426b-b3a7e4298b605de4f56edfff09169f1a.png)



```cpp
#define CEIL_DIV(M, N) (((M) + (N)-1) / (N))

dim3 gridDim(CEIL_DIV(M, 32), CEIL_DIV(N, 32), 1);
// 32 * 32 = 1024 thread per block
dim3 blockDim(32, 32, 1);
sgemm_naive<<<gridDim, blockDim>>>(M, N, K, alpha, A, B, beta, C);
```



## 内核实现

CUDA 代码是从单线程的视角编写的。在内核代码中，我们使用 `blockIdx` 和 `threadIdx`。这些变量的值会根据访问它们的线程而异。在我们的例子中，`threadIdx.x` 和 `threadIdx.y` 将根据线程在网格中的位置从 0 到 31 变化。同样，`blockIdx.x` 和 `blockIdx.y` 也将根据线程块在网格中的位置从 0 到 `CEIL_DIV(N, 32)` 或 `CEIL_DIV(M, 32)` 变化。

```cpp
__global__ void sgemm_naive_kernel(float *A, float *B, float *C, int M, int N, int K)
{
    const uint x = blockIdx.x * blockDim.x + threadIdx.x;
    const uint y = blockIdx.y * blockDim.y + threadIdx.y;
    if (x < M && y < N)
    {
        float sum = 0.0f;
        for (int i = 0; i < K; i++)
        {
            sum += A[x * K + i] * B[i * N + y];
        }
        C[x * N + y] = sum;
    }
}
```

![picture 1](https://cuda.keter.top/assets/images/6f55c7f9531e5efd955eab9a572ef5406733498bc0b50abed0e73985d88c840b-a41ab97d63a8f3d017bacaede20e8b5e.png)

## Warp和Block

线程束是SM中基本的执行单元，当一个网格被启动（网格被启动，等价于一个内核被启动，每个内核对应于自己的网格），网格中包含线程块，线程块被分配到某一个SM上以后，将分为多个线程束，每个线程束一般是32个线程（目前的GPU都是32个线程，但不保证未来还是32个）在一个线程束中，所有线程按照单指令多线程SIMT的方式执行，每一步执行相同的指令，但是处理的数据为私有的数据，下图反应的就是逻辑，实际，和硬件的图形化

![img](https://face2ai.com/CUDA-F-3-2-%E7%90%86%E8%A7%A3%E7%BA%BF%E7%A8%8B%E6%9D%9F%E6%89%A7%E8%A1%8C%E7%9A%84%E6%9C%AC%E8%B4%A8-P1/3_10.png)

线程束和线程块，一个是硬件层面的线程集合，一个是逻辑层面的线程集合，我们编程时为了程序正确，必须从逻辑层面计算清楚，但是为了得到更快的程序，硬件层面是我们应该注意的。

```
if (con)
{
    //do something
}
else
{
    //do something
}
```

假设这段代码是核函数的一部分，那么当一个线程束的32个线程执行这段代码的时候，如果其中16个执行if中的代码段，而另外16个执行else中的代码块，同一个线程束中的线程，执行不同的指令，这叫做**线程束的分化**。

因为线程束分化导致的性能下降就应该用线程束的方法解决，根本思路是避免同一个线程束内的线程分化，而让我们能控制线程束内线程行为的原因是线程块中线程分配到线程束是有规律的而不是随机的。这就使得我们根据线程编号来设计分支是可以的，补充说明下，当一个线程束中所有的线程都执行if或者，都执行else时，不存在性能下降；只有当线程束内有分歧产生分支的时候，性能才会急剧下降。
线程束内的线程是可以被我们控制的，那么我们就把都执行if的线程塞到一个线程束中，或者让一个线程束中的线程都执行if，另外线程都执行else的这种方式可以将效率提高很多。

```
// 下面这个kernel可以产生一个比较低效的分支：
__global__ void mathKernel1(float *c)
{
	int tid = blockIdx.x* blockDim.x + threadIdx.x;

	float a = 0.0;
	float b = 0.0;
	if (tid % 2 == 0)
	{
		a = 100.0f;
	}
	else
	{
		b = 200.0f;
	}
	c[tid] = a + b;
}

// 高效
// 第一个线程束内的线程编号tid从0到31，tid/warpSize都等于0，那么就都执行if语句。
// 第二个线程束内的线程编号tid从32到63，tid/warpSize都等于1，执行else
// 线程束内没有分支，效率较高。
__global__ void mathKernel2(float *c)
{
	int tid = blockIdx.x* blockDim.x + threadIdx.x;
	float a = 0.0;
	float b = 0.0;
	if ((tid/warpSize) % 2 == 0)
	{
		a = 100.0f;
	}
	else
	{
		b = 200.0f;
	}
	c[tid] = a + b;
}
```



## SharedMem

在全局内存之外，GPU 还有一块位于芯片上的较小区域，被称为共享内存（SMEM）。每个 SM（流多处理器）都配备了一块共享内存。

![picture 0](https://cuda.keter.top/assets/images/264915564b04781951d36d7d8527b418bbe0fea3a3969563a639f6575c1febd5-adf32b636af54b74924dbbf8f2d2fb5d.png)

上图中SM-0表示一个Block，蓝色底色表示一个Grid

从逻辑上看，共享内存在各个块之间进行了分区。这意味着一个线程可以通过共享内存块与同一块内的其他线程进行通信。共享内存的大小是可配置的，可以通过权衡以获得更大的共享内存而减小 L1 缓存的大小。

### 共享内存的使用

对于这个新的内核，我们将 A 和 B 的全局内存一块加载到共享内存中。接着，我们将在这两块上尽可能多地执行计算。这样做的好处是，我们可以**减少对全局内存的访问次数**，因为共享内存的访问速度比全局内存快得多。

计算的流程如下图所示，可以看到我们将 A 和 B 的一块加载到共享内存中，然后在共享内存中进行计算。

![picture 1](https://cuda.keter.top/assets/images/b99194dc785674eb6347c91f3b30e150d29fc238e2c63332641d9c55a205fd8f-ce2c38eb987040be927b813243edb690.png)

在内核中加入了共享内存的使用。每一个线程负责计算 C 中的一个元素。

以下是代码的重要部分，其中变量名称对应上面的图表：



```c++
__global__ void MatmulSharedStaticKernel(float *M_device, float *N_device,
                                         float *P_device, int A, int B, int C) {
  // 注意：这个BLOCKSIZE的大小应该和设置的blockSize大小保持一致
  __shared__ float M_deviceShared[BLOCKSIZE][BLOCKSIZE];
  __shared__ float N_deviceShared[BLOCKSIZE][BLOCKSIZE];

  int col = blockIdx.x * blockDim.x + threadIdx.x;  // 列索引 （0--> C-1）
  int row = blockIdx.y * blockDim.y + threadIdx.y;  // 行索引 （0--> A-1）

  float P_element = 0.;
  int ty = threadIdx.y;
  int tx = threadIdx.x;

  // 搬运数据到Shared mem，搬运多少数据呢？依次搬运的应该是 2 * BLOCKSIZE * BLOCKSIZE
  for (int m = 0; m < (B + BLOCKSIZE - 1) / BLOCKSIZE; m++) {
    int M_col = m * BLOCKSIZE + tx;
    if (row < A && M_col < B) {
      M_deviceShared[ty][tx] = M_device[row * B + M_col];
    } else {
      M_deviceShared[ty][tx] = 0.0;
    }
    // Load N tile (B x C matrix)
    int N_row = m * BLOCKSIZE + ty;
    if (N_row < B && col < C) {
      N_deviceShared[ty][tx] = N_device[N_row * C + col];
    } else {
      N_deviceShared[ty][tx] = 0.0;
    }

    __syncthreads();

    // Compute partial product
    for (int k = 0; k < BLOCKSIZE; ++k) {
      P_element += M_deviceShared[ty][k] * N_deviceShared[k][tx];
    }
    __syncthreads();
  }
  if (row < A && col < C) {
    P_device[row * C + col] = P_element;
  }
}

```

### Bank Conflict

> 为了获得高的内存带宽，共享内存在物理上被分为 32 个同样宽度的、能被同时访问的内存 bank。共享内存中每连续的 128 字节的内容分摊到 32 个 bank 的同一层中。bank 是共享内存的最小单元。

在CUDA架构中，共享内存被划分为多个大小相等的存储体**(bank)**，这些bank可以并行访问。**当多个线程同时访问同一个bank的不同地址时，就会发生bank conflict，导致这些访问必须串行执行，从而降低性能。**

同一个 Block 的线程会共享一块共享内存，**Bank conflict 是指一个 warp 内的多个线程同时访问同一个 bank 的不同地址，那么它们的访问就会被串行化，从而降低性能**。

![picture 1](https://cuda.keter.top/assets/images/ef322be7c3e5b6b9be69d2b90e88083f50569a58a97129f348e483b946ab4edf-4add931994c0dd899a0ab4c58db54114.png)

上图中T-0和T-1同时访问1号bank中的数据（红色的1 和 33），此时就发生了BankConflict

#### 解决





## 参考

https://cuda.keter.top/

https://github.com/PaddleJitLab/CUDATutorial/tree/develop/docs

https://face2ai.com/program-blog/#GPU%E7%BC%96%E7%A8%8B%EF%BC%88CUDA%EF%BC%89