#include <stdio.h>
#include <cuda_runtime.h>
#include "utils.hpp"
#include "timer.hpp"
#include "matmul.hpp"

int seed;
int main() {
  Timer timer;

  int width = 1 << 6;
  int min = 0, max = 1;
  int size = width * width;
  int block_size = 32;

  float* h_matM = new float[size];
  float* h_matN = new float[size];
  float* h_matP = new float[size];
  float* d_matP = new float[size];

  seed = 1;
  initMatrix(h_matM, size, min, max, seed);
  seed++;
  initMatrix(h_matN, size, min, max, seed);


  timer.start();
  MatMulOnHost(h_matM, h_matN, h_matP, width);
  timer.stop();
  timer.duration<Timer::ms>("matmul in cpu");

  /* GPU warmup */
  timer.start();
  MatMulOnDevice(h_matM, h_matN, d_matP, width, block_size);
  timer.stop();
  timer.duration<Timer::ms>("matmul in gpu(warmup)");
  for (int i = 0; i < width; ++i) {
    for (int j = 0; j < width; ++j) {
        printf("%6.2f \t", d_matP[i * width + j]);
    }
  }

  delete[] d_matP;
  delete[] h_matP;
  delete[] h_matN;
  delete[] h_matM;
  return 0;
}