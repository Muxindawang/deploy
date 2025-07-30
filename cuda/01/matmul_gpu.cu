#include <cuda_runtime.h>
#include <cuda.h>
#include <stdio.h>
#include "matmul.hpp"

__global__ void MatmulKernel(float *M_device, float *N_device, float *P_device, int width) {
  /* 
    我们设定每一个thread负责P中的一个坐标的matmul
    一个坐标的乘积需要遍历width元素乘积（一行乘一列）
    所以一共有width * width个thread并行处理P的计算
  */

  // x 和 y 分别表示输出矩阵P中的列和行的index
  // p[y][x] = M[y, :] * N[: ,x] 这是矩阵的计算方式
  int x = blockIdx.x * blockDim.x + threadIdx.x;
  int y = blockIdx.y * blockDim.y + threadIdx.y;

  float P_element = 0;
  // 如何取M和N的整行和整列? k表示一行（一列）元素的index 从0到width- 1
  for (int k = 0; k < width; k++) {
    float M_element = M_device[y * width + k];
    float N_element = N_device[k * width + x];
    P_element += M_element * N_element;
  }

  P_device[y * width + x] = P_element;
}


void MatMulOnDevice(float *M_host, float* N_host, float* P_host, int width, int block_size) {
  int size = width * width * sizeof(float);

  float *M_device, *N_device;

  cudaMalloc(&M_device, size);
  cudaMalloc(&N_device, size);

  cudaMemcpy(M_device, M_host, size, cudaMemcpyKind::cudaMemcpyHostToDevice);
  cudaMemcpy(N_device, N_host, size, cudaMemcpyKind::cudaMemcpyHostToDevice);

  float* P_device;
  cudaMalloc(&P_device, size);

  dim3 dimBlock(block_size, block_size);
  // dimGrid为什么设置这个维度？必须保证 dimGrid * dimBlock >= width 即一次性计算矩阵中的全部数据，如果设置的比width小，会导致矩阵右下角的数据不会被计算
  dim3 dimGrid(width / block_size, width / block_size);
  MatmulKernel <<<dimGrid, dimBlock>>>(M_device, N_device, P_device, width);

  cudaMemcpy(P_host, P_device, size, cudaMemcpyDeviceToHost);

  cudaFree(P_device);
  cudaFree(N_device);
  cudaFree(M_device);
}