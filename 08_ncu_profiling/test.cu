#include "cuda_runtime.h"
#include "device_launch_parameters.h"

#include <stdio.h>

__global__ void sum(int* a, int* b, int* c)
{
    int i = threadIdx.x;
    c[i] = a[i] + b[i];
}

int main()
{
    const int size = 1024;

    // Allocate memory on CPU
    int* a = (int*)malloc(sizeof(int) * size);
    int* b = (int*)malloc(sizeof(int) * size);
    int* c = (int*)malloc(sizeof(int) * size);

    // Allocate memory on GPU
    int* a_d, *b_d, *c_d;
    cudaMalloc((void**)&a_d, sizeof(int) * size);
    cudaMalloc((void**)&b_d, sizeof(int) * size);
    cudaMalloc((void**)&c_d, sizeof(int) * size);
    // kernal_name<<<num_of_blocks, num_of_threads_per_block>>>();
    // cudaDeviceSynchronize();

    // Fill array with values
    for (int i = 0; i < size; ++i)
    {
        a[i] = i;
        b[i] = size - i;
    }

    // Copy data from CPU to GPU
    cudaMemcpy(a_d, a, sizeof(int) * size, cudaMemcpyHostToDevice);
    cudaMemcpy(b_d, b, sizeof(int) * size, cudaMemcpyHostToDevice);

    // Calculate sum
    sum<<<1, 1024>>>(a_d ,b_d, c_d);
    cudaDeviceSynchronize();
    
    // Copy result from GPU tp CPU
    cudaMemcpy(c, c_d, sizeof(int) * size, cudaMemcpyDeviceToHost);

    // Print result
    // printf("Execution completed\n");
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