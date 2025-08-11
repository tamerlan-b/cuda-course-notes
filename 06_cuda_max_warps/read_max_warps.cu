#include "cuda_runtime.h"
#include <stdio.h>

int main(){
    int device;
    cudaGetDevice(&device);
    cudaDeviceProp prop;

    cudaGetDeviceProperties(&prop, device);

    printf("1) Using cudaGetDeviceProperties()\n");
    printf("\tMax threads per SM: %d\n", prop.maxThreadsPerMultiProcessor);
    printf("\tMax warps per SM: %d\n", prop.maxThreadsPerMultiProcessor / 32);

    printf("2) Using cudaGetDeviceProperties()\n");
    int max_threads_per_sm = 0;
    cudaDeviceGetAttribute(&max_threads_per_sm, cudaDevAttrMaxThreadsPerMultiProcessor, device);
    printf("\tMax threads per SM: %d\n", max_threads_per_sm);
    printf("\tMax warps per SM: %d\n", max_threads_per_sm / 32);
    return 0;
}