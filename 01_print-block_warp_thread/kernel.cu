#include "cuda_runtime.h"
#include "device_launch_parameters.h"

#include <stdio.h>

__global__ void test01()
{
    int warp_id = threadIdx.x / 32;
    printf("Block id: %d --- Warp id: %d --- Thread id: %d\n", blockIdx.x, warp_id, threadIdx.x);
}

int main()
{
    // kernal_name<<<num_of_blocks, num_of_threads_per_block>>>();
    test01<<<2,64>>>();
    cudaDeviceSynchronize();
    return 0;
}