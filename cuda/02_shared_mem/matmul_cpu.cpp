#include "matmul.hpp"

void MatmulOnHost(float *M, float *N, float *P, int A, int B, int C){
  for (int i = 0; i < A; i++) {
    for(int j = 0; j < C; j++) {
      float sum = 0;
      for (int k = 0; k < B; k++) {
        sum += M[i * B + k] * N[k * C + j];
      }
      P[i * C + j] = sum;
    }
  }
}

void MataddOnHost(float *M, float *N, float *P, int width){
    for (int i = 0; i < width; i ++)
        for (int j = 0; j < width; j ++){
            int idx = j * width + i;
            P[idx] = M[idx] + N[idx];
        }
}
