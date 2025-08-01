#include <cuda_runtime.h>
#include <stdio.h>
#include "reduce.hpp"

const int BLOCKSIZE = 4;
__global__ void reduce_naive_kernel(int* arr, int* out, int len) {
  __shared__ int smem[BLOCKSIZE];

  int idx = blockIdx.x * blockDim.x + threadIdx.x;
  int tid = threadIdx.x;

  smem[tid] = (idx < len) ? arr[idx] : 0;
  __syncthreads();
  // 把block中后一半的线程的值加到前一半中，如此往复，直到全部加到第0位中
  for (int s = blockDim.x / 2; s > 0; s <<= 1) {
      if (tid < s) smem[tid] += smem[tid + s];
      __syncthreads();
  }

  if (tid == 0) out[blockIdx.x] = smem[0];
}

const int len = 10;

int main() {
    int *arr = new int[len];
    int *out = new int[len];
    int *d_arr, *d_out;

    // 初始化数组
    for (int i = 0; i < len; i++) {
        arr[i] = i;
    }

    // 分配内存
    cudaMalloc((void **)&d_arr, sizeof(int) * len);
    cudaMalloc((void **)&d_out, sizeof(int) * len);

    // 拷贝数据到显存
    cudaMemcpy(d_arr, arr, sizeof(int) * len, cudaMemcpyHostToDevice);
    cudaMemcpy(d_out, out, sizeof(int) * len, cudaMemcpyHostToDevice);

    // 计算 block 和 grid 的大小
    int gridsize = (len + BLOCKSIZE - 1) / BLOCKSIZE;

    // 调用 kernel 函数
    reduce_naive_kernel<<<gridsize, BLOCKSIZE>>>(d_arr, d_out, len);

    // 拷贝数据到内存
    cudaMemcpy(out, d_out, sizeof(int) * len, cudaMemcpyDeviceToHost);

    // 计算结果
    int sum = 0;
    // 注意是gridsize，不是blocksize
    // 因为每个block的第一个线程都会把自己的值写入到out中
    // gridsize是block的数量（结合图理解）
    for (int i = 0; i < gridsize; i++) {
        sum += out[i];
    }
    printf("sum = %d\n", sum);

    // 核对结果
    int sum2 = 0;
    for (int i = 0; i < len; i++) {
        sum2 += arr[i];
    }

    if (sum == sum2) {
        printf("success\n");
    } else {
        printf("failed\n");
    }

    // 释放内存
    cudaFree(d_arr);
    cudaFree(d_out);
    delete[] arr;
    delete[] out;
    return 0;
}