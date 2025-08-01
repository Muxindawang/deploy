#include <iostream>

#include "cuda_runtime_api.h"
#include "stdio.h"
#include "utils.hpp"

__global__ void resize_nearest_BGR2RGB_kernel(uint8_t* tar, uint8_t* src,
                                              int tarW, int tarH, int srcW,
                                              int srcH, float scaled_w,
                                              float scaled_h) {
  int col = blockIdx.x * blockDim.x + threadIdx.x;
  int row = blockIdx.y * blockDim.y + threadIdx.y;

  int src_col = round((float)col * scaled_w);
  int src_row = round((float)row * scaled_h);

  if (src_col >= 0 && src_col < srcW && src_row >= 0 && src_row < srcH) {
    int tar_idx = (row * tarW + col) * 3;
    int src_idx = (src_row * srcW + src_col) * 3;
    tar[tar_idx + 0] = src[src_idx + 2];
    tar[tar_idx + 1] = src[src_idx + 1];
    tar[tar_idx + 2] = src[src_idx + 0];
  }
}

void resize_bilinear_gpu(uint8_t* d_tar, uint8_t* d_src, int tarW, int tarH,
                         int srcW, int srcH, int tactis) {
  dim3 dimBlock(16, 16, 1);
  dim3 dimGrid((tarW + 16 - 1) / 16, (tarH + 16 - 1) / 16, 1);

  float scaled_h = (float)srcH / tarH;
  float scaled_w = (float)srcW / tarW;
  float scale = std::fmax(scaled_h, scaled_w);

  if (tactis > 1) {
    scaled_h = scale;
    scaled_w = scale;
  }

  resize_nearest_BGR2RGB_kernel<<<dimGrid, dimBlock>>>(
      d_tar, d_src, tarW, tarH, srcW, srcH, scaled_w, scaled_h);
}