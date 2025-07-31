#ifndef __MATMUL_HPP__
#define __MATMUL_HPP__

void MatmulOnDevice(float *M_host, float *N_host, float* P_host, int A, int B, int C, int blockSize);
void MatmulSharedOnDevice(float *M_host, float *N_host, float* P_host, int A, int B, int C, int blockSize, bool staticMem);
extern void MatmulOnHost(float *M_host, float *N_host, float* P_host, int A, int B, int C);

#endif //__MATMUL_HPP__
