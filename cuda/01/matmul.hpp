#pragma once

void MatMulOnDevice(float *M_host, float* N_host, float* P_host, int width, int block_size);
void MatMulOnHost(float *M_host, float* N_host, float* P_host, int width);