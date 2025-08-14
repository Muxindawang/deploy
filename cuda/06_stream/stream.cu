#include <cuda_runtime.h>
#include <utils.hpp>

#define MAX_ITER 100000
#define SIZE 32

__global__ void SleepKernel(int64_t num_cycles) {
  int64_t cycles = 0;
  int64_t start = clock64();
  while(cycles < num_cycles) {
    cycles = clock64() - start;
  }
}

void SleepSingleStream(float* src_host, float* tar_host, int width, 
      int block_size, int count) {
  int size = width * width * sizeof(float);
  float* src_device, *tar_device;

  CUDA_CHECK(cudaMalloc(&src_device, size));
  CUDA_CHECK(cudaMalloc(&tar_device, size));

  for (int i = 0; i < count; i++) {
    for (int j = 0; j < 1; j++) {
      CUDA_CHECK(cudaMemcpy(src_device, src_host, size, cudaMemcpyHostToDevice));
    }
    dim3 dimBlock(block_size, block_size);
    dim3 dimGrid(width / block_size, width / block_size);
    SleepKernel<<<dimGrid, dimBlock>>> (MAX_ITER);
    CUDA_CHECK(cudaMemcpy(src_host, src_device, size, cudaMemcpyDeviceToHost));
  }

  CUDA_CHECK(cudaDeviceSynchronize());

  cudaFree(tar_device);
  cudaFree(src_device);
}

/* n stream，处理一次memcpy，以及n个kernel */
void SleepMultiStream(float* src_host, float* tar_host, int width,
                      int blockSize, int count) {
  int size = width * width * sizeof(float);

  float* src_device;
  float* tar_device;

  CUDA_CHECK(cudaMalloc((void**)&src_device, size));
  CUDA_CHECK(cudaMalloc((void**)&tar_device, size));

  /* 先把所需要的stream创建出来 */
  cudaStream_t stream[count];
  for (int i = 0; i < count; i++) {
    CUDA_CHECK(cudaStreamCreate(&stream[i]));
  }

  for (int i = 0; i < count; i++) {
    for (int j = 0; j < 1; j++)
      CUDA_CHECK(cudaMemcpyAsync(src_device, src_host, size,
                                 cudaMemcpyHostToDevice, stream[i]));
    dim3 dimBlock(blockSize, blockSize);
    dim3 dimGrid(width / blockSize, width / blockSize);

    /* 这里面我们把参数写全了 <<<dimGrid, dimBlock, sMemSize, stream>>> */
    SleepKernel<<<dimGrid, dimBlock, 0, stream[i]>>>(MAX_ITER);
    CUDA_CHECK(cudaMemcpyAsync(src_host, src_device, size,
                               cudaMemcpyDeviceToHost, stream[i]));
  }

  CUDA_CHECK(cudaDeviceSynchronize());

  cudaFree(tar_device);
  cudaFree(src_device);

  for (int i = 0; i < count; i++) {
    // 使用完了以后不要忘记释放
    cudaStreamDestroy(stream[i]);
  }
}

/* n stream，处理一次memcpy，以及n个kernel */
