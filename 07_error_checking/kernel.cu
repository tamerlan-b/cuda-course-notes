#include "cuda_runtime.h"
#include "device_launch_parameters.h"
#include <cuda.h>

#include <stdio.h>


inline void gpuAssert(cudaError_t err_code, const char* file, int line, bool abort=true)
{
    if(err_code != cudaSuccess)
    {
        fprintf(stderr, "Cuda error: %s %s line %d\n", cudaGetErrorString(err_code), file, line);
        if(abort){exit(err_code);}
    }
}
// Error checking macro
#define cudaCheckError(res){ gpuAssert(res, __FILE__, __LINE__); }

inline void gpuKernelAssert(const char* file, int line, bool abort=true)
{
    cudaError_t err_code = cudaGetLastError();
    if(err_code != cudaSuccess)
    {
        fprintf(stderr, "Cuda kernel error: %s %s line %d\n", cudaGetErrorString(err_code), file, line);
        if(abort){exit(err_code);}
    }
}
#define cudaKernelCheckError(){ gpuKernelAssert(__FILE__, __LINE__); }


__global__ void sum(int* a, int* b, int* c, long n)
{
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    c[i] = a[i] + b[i];
}

int main()
{
    const long size = 1024LL * 1024 * 1024 * 20;

    // cudaError_t err;
    
    // Allocate memory on GPU
    int* a_d, *b_d, *c_d;
    cudaCheckError(cudaMalloc((void**)&a_d, sizeof(int) * size))    // Check synchronous error
    cudaCheckError(cudaMalloc((void**)&b_d, sizeof(int) * size))    // Check synchronous error
    cudaCheckError(cudaMalloc((void**)&c_d, sizeof(int) * size))    // Check synchronous error
    // kernal_name<<<num_of_blocks, num_of_threads_per_block>>>();
    // cudaDeviceSynchronize();
    
    // Allocate memory on CPU
    int* a = (int*)malloc(sizeof(int) * size);
    int* b = (int*)malloc(sizeof(int) * size);
    int* c = (int*)malloc(sizeof(int) * size);

    // Fill array with values
    for (int i = 0; i < size; ++i)
    {
        a[i] = i;
        b[i] = size - i;
    }

    // Copy data from CPU to GPU
    cudaMemcpy(a_d, a, sizeof(int) * size, cudaMemcpyHostToDevice);
    cudaMemcpy(b_d, b, sizeof(int) * size, cudaMemcpyHostToDevice);

    const int grid_num = 64;
    const int block_dim = 32;
    cudaEvent_t start, stop;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);
    // Calculate sum
    cudaEventRecord(start);
    
    sum<<<grid_num, block_dim>>>(a_d ,b_d, c_d, size);
    cudaKernelCheckError()  // Check asynchronous error

    cudaDeviceSynchronize();
    cudaEventRecord(stop);
    cudaEventSynchronize(stop);
    float milliseconds = 0.F;
    cudaEventElapsedTime(&milliseconds, start, stop);
    
    // Copy result from GPU tp CPU
    cudaMemcpy(c, c_d, sizeof(int) * size, cudaMemcpyDeviceToHost);

    printf("Execution time sum<<<%d,%d>>>(): %f ms\n", grid_num, block_dim, milliseconds);

    // Print result
    // printf("Execution completed\n");
    // for (int i = 0; i < size; ++i)
    // {
    //     printf("%d + %d = %d\n", a[i], b[i], c[i]);
    // }

    cudaEventDestroy(start);
    cudaEventDestroy(stop);

    // Free memory on GPU
    cudaFree(a_d);
    cudaFree(b_d);
    cudaFree(c_d);

    // Free memory on CPU
    free(a);
    free(b);
    free(c);
    return 0;
}