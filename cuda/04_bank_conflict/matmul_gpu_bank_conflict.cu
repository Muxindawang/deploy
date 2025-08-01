#include "cuda_runtime_api.h"
#include "utils.hpp"

#define BLOCKSIZE 16

/* 
    使用shared memory把计算一个tile所需要的数据分块存储到访问速度快的memory中
*/
__global__ void MatmulSharedStaticConflictKernel(float *M_device, float *N_device, float *P_device, int width){
    __shared__ float M_deviceShared[BLOCKSIZE][BLOCKSIZE];
    __shared__ float N_deviceShared[BLOCKSIZE][BLOCKSIZE];
    /* 
        对于x和y, 根据blockID, tile大小和threadID进行索引
    */
    int x = blockIdx.x * blockDim.x + threadIdx.x;
    int y = blockIdx.y * blockDim.y + threadIdx.y;

    float P_element = 0.0;

    int ty = threadIdx.y;
    int tx = threadIdx.x;
    /* 对于每一个P的元素，我们只需要循环遍历width / tile_width 次就okay了，这里有点绕，画图理解一下*/
    for (int m = 0; m < width / BLOCKSIZE; m ++) {
        /* 这里为了实现bank conflict, 把tx与tx的顺序颠倒，同时索引也改变了*/
        // 顺序颠倒了之后，多个线程同时访问的就是一列数据 共享内存中一列数据属于同一个bank
        M_deviceShared[tx][ty] = M_device[x * width + (m * BLOCKSIZE + ty)];
        N_deviceShared[tx][ty] = M_device[(m * BLOCKSIZE + tx)* width + y];
        __syncthreads();

        for (int k = 0; k < BLOCKSIZE; k ++) {
            P_element += M_deviceShared[tx][k] * N_deviceShared[k][ty];
        }
        __syncthreads();
    }

    /* 列优先 */
    P_device[x * width + y] = P_element;
}

__global__ void MatmulSharedDynamicConflictKernel(float *M_device, float *N_device, float *P_device, int width, int blockSize){
    /* 
        声明动态共享变量的时候需要加extern，同时需要是一维的 
        注意这里有个坑, 不能够像这样定义： 
            __shared__ float M_deviceShared[];
            __shared__ float N_deviceShared[];
        因为在cuda中定义动态共享变量的话，无论定义多少个他们的地址都是一样的。
        所以如果想要像上面这样使用的话，需要用两个指针分别指向shared memory的不同位置才行
    */

    extern __shared__ float deviceShared[];
    int stride = blockSize * blockSize;
    /* 
        对于x和y, 根据blockID, tile大小和threadID进行索引
    */
    int x = blockIdx.x * blockSize + threadIdx.x;
    int y = blockIdx.y * blockSize + threadIdx.y;

    float P_element = 0.0;

    int ty = threadIdx.y;
    int tx = threadIdx.x;
    /* 对于每一个P的元素，我们只需要循环遍历width / tile_width 次就okay了 */
    for (int m = 0; m < width / blockSize; m ++) {
        /* 这里为了实现bank conflict, 把tx与tx的顺序颠倒，同时索引也改变了*/
        deviceShared[tx * blockSize + ty]             = M_device[x * width + (m * blockSize + ty)];
        deviceShared[stride + (tx * blockSize + ty)]  = N_device[(m * blockSize + tx) * width + y];

        __syncthreads();

        for (int k = 0; k < blockSize; k ++) {
            P_element += deviceShared[tx * blockSize + k] * deviceShared[stride + (k * blockSize + ty)];
        }
        __syncthreads();
    }

    /* 列优先 */
    P_device[x * width + y] = P_element;
}

/*
    使用Tiling技术
    一个tile处理的就是block, 将一个矩阵分为多个小的tile，这些tile之间的执行独立，并且可以并行
*/
void MatmulSharedConflictOnDevice(float *M_host, float *N_host, float* P_host, int width, int blockSize, bool staticMem){
    /* 设置矩阵大小 */
    int size = width * width * sizeof(float);
    long int sMemSize = blockSize * blockSize * sizeof(float) * 2;

    /* 分配M, N在GPU上的空间*/
    float *M_device;
    float *N_device;
    CUDA_CHECK(cudaMalloc((void**)&M_device, size));
    CUDA_CHECK(cudaMalloc((void**)&N_device, size));

    /* 分配M, N拷贝到GPU上*/
    CUDA_CHECK(cudaMemcpy(M_device, M_host, size, cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(N_device, N_host, size, cudaMemcpyHostToDevice));

    /* 分配P在GPU上的空间*/
    float *P_device;
    CUDA_CHECK(cudaMalloc((void**)&P_device, size));;

    /* 调用kernel来进行matmul计算, 在这个例子中我们用的方案是：使用一个grid，一个grid里有width*width个线程 */
    dim3 dimBlock(blockSize, blockSize);
    dim3 dimGrid(width / blockSize, width / blockSize);
    if (staticMem) {
        MatmulSharedStaticConflictKernel <<<dimGrid, dimBlock>>> (M_device, N_device, P_device, width);
    } else {
        MatmulSharedDynamicConflictKernel <<<dimGrid, dimBlock, sMemSize, nullptr>>> (M_device, N_device, P_device, width, blockSize);
    }

    /* 将结果从device拷贝回host*/
    CUDA_CHECK(cudaMemcpy(P_host, P_device, size, cudaMemcpyDeviceToHost));
    CUDA_CHECK(cudaDeviceSynchronize());

    /* 注意要在synchronization结束之后排查kernel的错误 */
    LAST_KERNEL_CHECK(); 

    /* Free */
    cudaFree(P_device);
    cudaFree(N_device);
    cudaFree(M_device);
}
