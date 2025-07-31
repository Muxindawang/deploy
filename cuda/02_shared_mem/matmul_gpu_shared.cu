#include "cuda_runtime_api.h"
#include "utils.hpp"

#define BLOCKSIZE 16

/*
    使用shared memory把计算一个tile所需要的数据分块存储到访问速度快的memory中
*/
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

__global__ void MatmulSharedDynamicKernel(float *M_device, float *N_device,
                                          float *P_device, int A, int B, int C,
                                          int block_size) {
  /*
      声明动态共享变量的时候需要加extern，同时需要是一维的
      注意这里有个坑, 不能够像这样定义：
          __shared__ float M_deviceShared[];
          __shared__ float N_deviceShared[];
      因为在cuda中定义动态共享变量的话，无论定义多少个他们的地址都是一样的。
      所以如果想要像上面这样使用的话，需要用两个指针分别指向shared
     memory的不同位置才行
  */

  extern __shared__ float deviceShared[];
  int stride = block_size * block_size;
  /*
      对于x和y, 根据blockID, tile大小和threadID进行索引
  */
  int x = blockIdx.x * block_size + threadIdx.x;
  int y = blockIdx.y * block_size + threadIdx.y;

  float P_element = 0.0;

  int ty = threadIdx.y;
  int tx = threadIdx.x;
  /* 对于每一个P的元素，我们只需要循环遍历width / tile_width 次就okay了 */
  // 动态mem 不能用二维寻址 只能用一维的
  for (int m = 0; m < (B + BLOCKSIZE - 1) / BLOCKSIZE; m++) {
    deviceShared[ty * block_size + tx] = M_device[y * B + (m * block_size + tx)];
    deviceShared[stride + (ty * block_size + tx)] =
        N_device[(m * block_size + ty) * C + x];
    __syncthreads();

    for (int k = 0; k < block_size; k++) {
      P_element += deviceShared[ty * block_size + k] *
                   deviceShared[stride + (k * block_size + tx)];
    }
    __syncthreads();
  }

  if (y < A && x < C) {
    P_device[y * C + x] = P_element;
  }
}

/*
  使用Tiling技术 一个tile处理的就是block,
  将一个矩阵分为多个小的tile，这些tile之间的执行独立，并且可以并行
*/
void MatmulSharedOnDevice(float *M_host, float *N_host, float *P_host, int A,
                          int B, int C, int block_size, bool staticMem) {
  if (!M_host || !N_host || !P_host || A <= 0 || B <= 0 || C <= 0) {
    printf("Invalid input parameters!\n");
    return;
  }
  // 注意：这里的size需要 * sizeof(float)
  int M_size = A * B * sizeof(float), N_size = B * C * sizeof(float),
      P_size = A * C * sizeof(float);
  long int sMemSize = block_size * block_size * sizeof(float) * 2;

  /* 分配M, N在GPU上的空间*/
  float *M_device;
  float *N_device;
  CUDA_CHECK(cudaMalloc(&M_device, M_size));
  CUDA_CHECK(cudaMalloc(&N_device, N_size));

  /* 分配M, N拷贝到GPU上*/
  CUDA_CHECK(cudaMemcpy(M_device, M_host, M_size, cudaMemcpyHostToDevice));
  CUDA_CHECK(cudaMemcpy(N_device, N_host, N_size, cudaMemcpyHostToDevice));

  /* 分配P在GPU上的空间*/
  float *P_device;
  CUDA_CHECK(cudaMalloc(&P_device, P_size));
  ;

  dim3 dimBlock(block_size, block_size);
  dim3 dimGrid((C + block_size - 1) / block_size,
               (A + block_size - 1) / block_size);
  // 使用静态变量和不适用 执行的核函数是不一样的
  if (staticMem) {
    MatmulSharedStaticKernel<<<dimGrid, dimBlock>>>(M_device, N_device,
                                                    P_device, A, B, C);
  } else {
    MatmulSharedDynamicKernel<<<dimGrid, dimBlock, sMemSize, nullptr>>>(
        M_device, N_device, P_device, A, B, C, block_size);
  }

  /* 将结果从device拷贝回host*/
  CUDA_CHECK(cudaMemcpy(P_host, P_device, P_size, cudaMemcpyDeviceToHost));
  CUDA_CHECK(cudaDeviceSynchronize());

  /* 注意要在synchronization结束之后排查kernel的错误 */
  LAST_KERNEL_CHECK();

  /* Free */
  cudaFree(P_device);
  cudaFree(N_device);
  cudaFree(M_device);
}
