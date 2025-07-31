#include "cuda_runtime_api.h"
#include "stdio.h"
#include <iostream>

#include "utils.hpp"

/* matmul的函数实现*/
__global__ void MatmulKernel(float *M_device, float *N_device, float *P_device, int A, int B, int C){
  
  // // 每个线程负责 P 中的一个元素 P(row, col) = P[row*C + col]
  int col = blockIdx.x * blockDim.x + threadIdx.x;  // 列号 (0..C-1)
  int row = blockIdx.y * blockDim.y + threadIdx.y;  // 行号 (0..A-1)

  if (col < C && row < A) {
    float sum = 0.0f;
    for (int k = 0; k < B; ++k) {
      sum += M_device[row * B + k] * N_device[k * C + col];
    }
    P_device[row * C + col] = sum;
  }
}

/*

    这个实现的问题点：只有一个block
    因为只有一个block，并且又因为SM中的sp数量是有限的，所以不能够全部放下。想要全部放下的话需要缩小矩阵的大小
    有很多次读写，但具体的执行很少(两次读和一次写，一次计算)
    解决办法：使用tile
*/
void MatmulOnDevice(float *M_host, float *N_host, float* P_host, int A, int B, int C, int blockSize){
  size_t M_size = A * B * sizeof(float), N_size = B * C * sizeof(float), P_size = A * C * sizeof(float);
  float *M_device, *N_device;

  CUDA_CHECK(cudaMalloc(&M_device, M_size));
  CUDA_CHECK(cudaMalloc(&N_device, N_size));

  CUDA_CHECK(cudaMemcpy(M_device, M_host, M_size, cudaMemcpyHostToDevice));
  CUDA_CHECK(cudaMemcpy(N_device, N_host, N_size, cudaMemcpyHostToDevice));


  float* P_device;
  CUDA_CHECK(cudaMalloc(&P_device, P_size));
  CUDA_CHECK(cudaMemset(P_device, 0, P_size));  // 初始化设备内存为0

  dim3 dimBlock(blockSize, blockSize);
  dim3 dimGrid((C + blockSize - 1) / blockSize,  // 修正
                (A + blockSize - 1) / blockSize);  // 修正
  MatmulKernel <<<dimGrid, dimBlock>>> (M_device, N_device, P_device, A, B, C);

  CUDA_CHECK(cudaMemcpy(P_host, P_device, P_size, cudaMemcpyDeviceToHost));
  CUDA_CHECK(cudaDeviceSynchronize());

  /* 注意要在synchronization结束之后排查kernel的错误 */
  LAST_KERNEL_CHECK(); 

  /* Free */
  cudaFree(P_device);
  cudaFree(N_device);
  cudaFree(M_device);
}

