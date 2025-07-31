## 基本概念

Thread、Block、Grid 的概念：

- `Thread`: 一个 CUDA Kernel 可以被多个 threads 来执行
- `Block`: 多个 threads 会组成一个 Block，而同一个 block 中的 threads 可以同步，也可以通过 shared memory 通信
- `Grid`: 多个 blocks 可以组成一个 Grid

其中，一个 Grid 可以包含多个 Blocks。Blocks 的分布方式可以是一维的，二维，三维的；Block 包含多个 Threads，Threads 的分布方式也可以是一维，二维，三维的。



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
先找到属于哪一个Block，在再该Block查找是哪个thread

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





## 参考

https://cuda.keter.top/

https://face2ai.com/program-blog/#GPU%E7%BC%96%E7%A8%8B%EF%BC%88CUDA%EF%BC%89