#include "matmul.hpp"

void MatMulOnHost(float *M, float* N, float* P, int width) {
  // i 表示第一个矩阵M的第i行
  // j 表示第二个矩阵N的第j行
  // k 表示元素的index
  // 那么结果P的第i行第j个元素就等于 第一个矩阵的第i行 乘 第二个矩阵的第j列
  for (int i = 0; i < width; i++) {
    for (int j = 0; j < width; j++) {
      float sum = 0;
      for (int k = 0; k < width; k++) {
        float a = M[i * width + k];
        float b = N[k * width + j];
        sum += a * b;
      }
      P[i * width + j] = sum;
    }
  }
}