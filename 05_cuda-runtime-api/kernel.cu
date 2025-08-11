#include <stdio.h>
#include <cuda_runtime_api.h>

int main()
{
    int num_devices;
    cudaGetDeviceCount(&num_devices);
    for(int i = 0; i < num_devices; ++i)
    {
        cudaDeviceProp prop;
        cudaGetDeviceProperties(&prop, i);
        printf("Device number: %d\n", i);
        printf("Device name: %s\n", prop.name);
        printf("Memory Clock Rate (KHz): %d\n", prop.memoryClockRate);
        printf("Memory Bus Width (bits): %d\n", prop.memoryBusWidth);
        printf("Peak Memory Bandwidth (GB/s): %f\n", 2.0*prop.memoryClockRate*(prop.memoryBusWidth/8.)/1.0e6);
        printf("Total Global Memory: %f (Gb)\n", prop.totalGlobalMem / 1024. / 1024. / 1024.);
        printf("Compute capability: %d.%d\n", prop.major, prop.minor);
        printf("Number of SMs: %d\n", prop.multiProcessorCount);
        printf("Max threads per block: %d\n", prop.maxThreadsPerBlock);
        printf("Max threads dimensions: x = %d, y = %d, z = %d\n", prop.maxThreadsDim[0], prop.maxThreadsDim[1], prop.maxThreadsDim[2]);
        printf("Max grid dimensions: x = %d, y = %d, z = %d\n", prop.maxGridSize[0], prop.maxGridSize[1], prop.maxGridSize[2]);
    }
    
    return 0;
}