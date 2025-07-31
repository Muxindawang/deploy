#include <stdio.h>
#include <cuda_runtime.h>

#include "utils.hpp"
#include "timer.hpp"
#include "matmul.hpp"

int seed;
int main() {
  Timer timer;

  int A = 1 << 10, B = 1 << 8, C = 1 << 10;
  int min = 0, max = 10;

  int block_size = 16;
  bool statMem  = true;

  float* h_matM = new float[A * B];
  float* h_matN = new float[B * C];
  float* h_matP = new float[A * C];
  float* d_matP = new float[A * C];
  char str[100];
  seed = 1;
  initMatrix(h_matM, A * B, min, max, seed);
  seed++;
  initMatrix(h_matN, B * C, min, max, seed);
    
  LOG("Input M size is %d x %d", A, B);
  LOG("Input N size is %d x %d", B, C);

  LOG("Input M: ");
  // for (int i = 0; i < A; i++) {
  //   for (int j = 0; j < B; j++) {
  //     printf("%6.2f,", h_matM[i * B + j]);
  //   }
  //   printf("\n");
  // }
  LOG("Input N: ");
  // for (int i = 0; i < B; i++) {
  //   for (int j = 0; j < C; j++) {
  //     printf("%6.2f,", h_matN[i * C + j]);
  //   }
  //   printf("\n");
  // }

  timer.start_gpu();
  MatmulOnHost(h_matM, h_matN, h_matP, A, B, C);
  timer.stop_gpu();
  timer.duration_gpu("matmul in cpu");
  

  /* GPU warmup */
  timer.start_gpu();
  MatmulOnDevice(h_matM, h_matN, h_matP, A, B, C, block_size);
  timer.stop_gpu();
  timer.duration_gpu("matmul in gpu(warmup)");

/* GPU general implementation <<<256, 16>>>*/
  timer.start_gpu();
  MatmulOnDevice(h_matM, h_matN, d_matP, A, B, C, block_size);
  timer.stop_gpu();

  std::sprintf(str, "matmul in gpu(without shared memory)<<<%d, %d, %d %d>>>", 
            (C + block_size - 1) / block_size, (A + block_size - 1) / block_size, block_size, block_size);
  timer.duration_gpu(str);
  compareMat(h_matP, d_matP, A * C);


  // /* GPU general implementation <<<256, 16>>>*/
  timer.start_gpu();
  MatmulSharedOnDevice(h_matM, h_matN, d_matP, A, B, C, block_size, statMem);
  timer.stop_gpu();
  std::sprintf(str, "matmul in gpu(with shared memory(static))<<<%d, %d>>>", C / block_size, block_size);
  timer.duration_gpu(str);
  compareMat(h_matP, d_matP, A * C);

  statMem = false;
  timer.start_gpu();
  MatmulSharedOnDevice(h_matM, h_matN, d_matP, A, B, C, block_size, statMem);
  timer.stop_gpu();
  std::sprintf(str, "matmul in gpu(with shared memory(dynamic))<<<%d, %d>>>", B / block_size, block_size);
  timer.duration_gpu(str);
  compareMat(h_matP, d_matP, A * C);
  // for (int i = 0; i < A; ++i) {
  //   for (int j = 0; j < C; ++j) {
  //       printf("%6.4f \t", h_matP[i * C + j]);
  //   }
  //   printf("\n");
  // }

  LOG("Device");
  // for (int i = 0; i < A; ++i) {
  //   for (int j = 0; j < C; ++j) {
  //       printf("%6.4f \t", d_matP[i * C + j]);
  //   }
  //   printf("\n");
  // }

  delete[] d_matP;
  delete[] h_matP;
  delete[] h_matN;
  delete[] h_matM;
  return 0;
}
