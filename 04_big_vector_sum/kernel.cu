#include "cuda_runtime.h"
#include "device_launch_parameters.h"
#include <cuda.h>

#include <stdio.h>

__global__ void sum(int* a, int* b, int* c, int n)
{
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    c[i] = a[i] + b[i];
}


int main()
{
    const int size = 1024*1024*1024;    // total number of elements in vector

    const int chunk_size = 1024*1024*128;   // elemnts per chuck, adjust based on availablewe host RAM

    static_assert(chunk_size <= size, "Chunk should be smaller of equal to the total size");

    // Allocate memory on CPU
    int* a = (int*)malloc(sizeof(int) * chunk_size);
    int* b = (int*)malloc(sizeof(int) * chunk_size);
    int* c = (int*)malloc(sizeof(int) * chunk_size);

    // Allocate memory on GPU
    int* a_d, *b_d, *c_d;
    cudaMalloc((void**)&a_d, sizeof(int) * chunk_size);
    cudaMalloc((void**)&b_d, sizeof(int) * chunk_size);
    cudaMalloc((void**)&c_d, sizeof(int) * chunk_size);

    const int block_dim = 1024; // num threads per block
    const int grid_num = (chunk_size + block_dim - 1) / block_dim;  // num blocks
    for (int offset = 0; offset < size; offset+=chunk_size)
    {
        const int current_chunk_size = (size - offset) < chunk_size ? (size - offset) : chunk_size;
        // Fill array with values
        for (int i = 0; i < current_chunk_size; ++i)
        {
            a[i] = i;
            b[i] = current_chunk_size - i;
        }
    
        // Copy data from CPU to GPU
        cudaMemcpy(a_d, a, sizeof(int) * current_chunk_size, cudaMemcpyHostToDevice);
        cudaMemcpy(b_d, b, sizeof(int) * current_chunk_size, cudaMemcpyHostToDevice);
    
        // Calculate sum
        sum<<<grid_num, block_dim>>>(a_d ,b_d, c_d, current_chunk_size);
        cudaDeviceSynchronize();
        printf("Execution of %d completed\n", offset);
        
        // Copy result from GPU tp CPU
        cudaMemcpy(c, c_d, sizeof(int) * current_chunk_size, cudaMemcpyDeviceToHost);
    
    }

    // Print result
    // for (int i = 0; i < size; ++i)
    // {
    //     printf("%d + %d = %d\n", a[i], b[i], c[i]);
    // }

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