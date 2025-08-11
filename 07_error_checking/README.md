### Проверка ошибок CUDA с использованием API

Компиляция:
```bash
nvcc kernel.cu -o kernel
```

#### Основы обработки ошибок

Ошибки могут возникать при вызове синхронных и асинхронных функций CUDA.  

1. Обработка ошибок синхронных функций
Синхронные функций возвращают код ошибки и их можно обработать следующим образом:  
```c++
#include "cuda_runtime.h"   // cudaError_t, cudaSuccess, cudaMalloc(...), cudaGetErrorString(...)
#include <stdio.h>  // fprintf(...)
...
int* a_d;
int size = 1024;
cudaError_t err_code = cudaMalloc((void**)&a_d, sizeof(int) * size)
if(err_code != cudaSuccess)
{
    fprintf(stderr, "Cuda error: %s\n", cudaGetErrorString(err_code));
}
...
```

2. Обработка ошибок асинхронных функций

Асинхронные функций не возвращают код ошибки поэтому их нужно обрабатывать иначе:  
```c++
#include "cuda_runtime.h"   // cudaError_t, cudaSuccess, cudaGetLastError(), cudaGetErrorString(...)
#include <stdio.h>  // fprintf(...)
...
__global__ void some_kernel(int* a)
{
    ...
}
...
int* a_d;
some_kernel<<<64, 32>>>(a_d);
cudaError_t err_code = cudaGetLastError();
if(err_code != cudaSuccess)
{
    fprintf(stderr, "Cuda kernel error: %s\n", cudaGetErrorString(err_code));
}
```


#### Удобные макросы для обработки ошибок

**Макрос для обработки синхронных ошибок:**  
```c++
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
```

Пример использования:  
```c++
int* a_d;
int size = 1024;
cudaCheckError(cudaMalloc((void**)&a_d, sizeof(int) * size))
```

**Макрос для обработки асинхронных ошибок:**  
```c++
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
```

Пример использования:  
```c++
some_kernel<<<64, 32>>>(a_d);
cudaKernelCheckError()
```